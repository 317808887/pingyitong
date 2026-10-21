"""应用编排。

负责程序从启动到退出的完整生命周期：装配各层组件、连接信号、
处理全局热键与显示器变化、以及有序退出。

分层装配顺序（自下而上，与开发文档一、的五层架构一致）::

    基础组件层   ScreenCapturer / OcrEngine / ModelPool / OverlayManager
    核心调度层   ResourceMonitor / ModelScheduler / CoordinateMapper / Pipeline
    业务功能层   TranslationService / Glossary / HistoryService / HotkeyService
    UI 交互层    TrayIcon / MainPanel / FloatingBar / RegionSelector / ToastManager

装配完成后，UI 层的任何入口（托盘菜单、悬浮条、热键、面板）都只会
调用本类暴露的动作方法，不直接触达下层组件——这样"同一动作的多个入口"
天然保持状态一致。
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from copy import deepcopy
from dataclasses import asdict, replace

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QIcon
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon

from ..components.capture import FrameDiffer, ScreenCapturer
from ..components.capture.geometry import Rect
from ..components.llm import ModelPool, PersistentCacheStore, TranslationCache
from ..components.ocr import OcrEngine
from ..components.render import OverlayManager
from ..config import SettingsManager
from ..constants import (
    APP_NAME,
    APP_NAME_EN,
    APP_ORG,
    APP_USER_MODEL_ID,
    ICON_APP_ICO,
    ICON_APP_PNG,
    MODEL_ACCURATE,
    MODEL_FAST,
    MODEL_LABELS,
)
from ..core import CoordinateMapper, ModelScheduler, Pipeline, ResourceMonitor
from ..services import (
    ACTION_COPY_RESULT,
    ACTION_SWITCH_MODEL,
    ACTION_TOGGLE_OVERLAY,
    ACTION_TOGGLE_TRANSLATION,
    Glossary,
    HistoryService,
    HotkeyService,
    TextPreprocessor,
    TranslationService,
    default_bindings,
)
from ..utils import (
    SingleInstance,
    get_logger,
    icon_path,
    install_exception_hooks,
    paths,
    reap_stale_servers,
)
from ..utils.logger import setup_logging
from ..version import version_string
from .design import Theme, apply_reduced_motion, build_theme, stylesheet
from .floating_bar import FloatingBar
from .history_dialog import HistoryDialog
from .main_panel import MainPanel
from .region_selector import RegionSelector
from .tray_icon import TrayIcon
from .widgets import ToastKind, ToastManager
from .widgets.tooltip import TooltipController
from .widgets.transcript_panel import TranscriptPanel

logger = get_logger(__name__)

#: 运行状态巡检间隔（毫秒）：模型空闲卸载、缓存清理、历史裁剪
_MAINTENANCE_INTERVAL_MS = 15_000


class _SerializedOcrEngine(OcrEngine):
    """Serialize background warm-up with live OCR's mutable backend state."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._inference_lock = threading.Lock()

    def warm_up(self) -> bool:
        with self._inference_lock:
            return super().warm_up()

    def recognize_frame(self, frame):
        with self._inference_lock:
            return super().recognize_frame(frame)


class Application(QObject):
    """应用编排器。

    由 :func:`run` 创建并持有；所有 UI 组件通过它访问下层能力。
    """

    #: 热键触发（从热键线程转发到主线程）
    hotkey_triggered = pyqtSignal(str)
    #: 流水线状态更新
    status_updated = pyqtSignal(object)
    #: 模型预加载进度（从预加载线程转发到主线程）
    preload_progress = pyqtSignal(object)
    #: 模型预加载完成
    preload_finished = pyqtSignal(object)
    #: 后台维护任务完成（用于在主线程刷新界面信息）
    maintenance_finished = pyqtSignal()
    #: 未捕获异常报告（可能来自后台线程，经队列连接转发到主线程）
    exception_reported = pyqtSignal(str, str, str)
    #: 字幕/网页预热进度（后台 worker 转发到 UI 线程）
    prefetch_progress = pyqtSignal(str, int, int)

    def __init__(self, qt_app: QApplication, portable: bool = False) -> None:
        super().__init__()
        self._qt_app = qt_app
        from ..utils.capture_exclusion import CaptureExclusionFilter

        # Install before creating overlays: capturing their translated pixels
        # causes repeated translation, mixed text and an unbounded work loop.
        self._capture_exclusion = CaptureExclusionFilter(qt_app)
        self._single_instance: SingleInstance | None = None
        self._maintenance_timer: QTimer | None = None
        #: 后台维护是否正在进行（避免慢磁盘时线程堆积）
        self._maintenance_running = False
        #: Only sweep orphaned servers before this instance can start inference.
        self._stale_servers_reaped = False
        self._pending_reaped_server_count = 0
        self._ocr_warm_lock = threading.Lock()
        self._ocr_warm_running = False
        self._ocr_warmed_backend: object | None = None
        #: 最近一次同步到界面的模型，避免状态轮询时重复刷新
        self._last_synced_model: str = ""
        #: 最近一次应用到流水线的翻译语言方向，用于检测语言切换
        #: （切换后屏幕内容未变，差分会认为"没有变化"而照旧跳过，
        #: 必须强制下一帧整帧识别，译文才会按新语言重新产出）
        self._last_translation_direction: tuple[str, str] | None = None
        #: 在线 AI 设置的快照；Key / 地址 / 模型切换后不能继续显示旧路由的译文。
        self._last_api_settings: dict[str, object] | None = None
        self._subtitle_lines_lock = threading.Lock()
        self._subtitle_lines: dict[str, dict[int, dict]] = {}
        self._prefetch_shutdown = False

        self._build_layers()
        self._build_ui()
        self._connect_signals()
        self._capture_exclusion.windows_changed.connect(
            self._on_owned_windows_changed, Qt.ConnectionType.QueuedConnection
        )
        self._push_ignore_regions()

        logger.info("%s %s 初始化完成", APP_NAME, version_string())

    # ------------------------------------------------------------------
    # 装配：基础组件层 → 业务功能层
    # ------------------------------------------------------------------

    def _build_layers(self) -> None:
        """装配各层组件。"""
        # ---- 配置 ----
        self.settings = SettingsManager()
        config = self.settings.load()

        # ---- 主题 ----
        self.theme: Theme = build_theme(config.ui.theme)
        self._apply_reduced_motion_setting()

        # ---- 基础组件层 ----
        self.capturer = ScreenCapturer()
        self.differ = FrameDiffer(
            pixel_threshold=config.capture.diff_pixel_threshold,
            min_area_ratio=config.capture.diff_min_area_ratio,
            block_size=config.capture.diff_block_size,
        )
        self.ocr_engine = _SerializedOcrEngine(
            upscale_factor=config.ocr.upscale_factor,
            enable_preprocess=config.ocr.enable_preprocess,
            min_confidence=config.ocr.min_confidence,
            retain_uncertain=config.ocr.retain_uncertain,
            min_box_height=config.ocr.min_box_height,
            box_padding=config.ocr.box_padding,
            max_blocks_per_frame=config.ocr.max_blocks_per_frame,
            threads=config.ocr.threads,
            source_language=config.translation.source_language,
        )
        self.model_pool = ModelPool(
            config.model, idle_timeout=config.schedule.accurate_idle_timeout
        )
        # Prefetch workers can start during layer construction. Sweep before
        # those workers or startup hotkeys can launch our own local servers.
        self._reap_stale_servers()
        # 在线 AI 配置在启动时同步应用一次：用户上次会话启用了在线模型时，
        # 首次翻译就要走云端路径，而不是等第一次配置变更信号
        self.model_pool.apply_api_config(
            replace(config.api, enabled=False) if config.translation.local_only else config.api
        )
        # 推理缓存：启用时接 SQLite 持久层，跨会话重复文案免二次推理
        cache_store: PersistentCacheStore | None = None
        if config.model.enable_cache:
            cache_store = PersistentCacheStore(paths.translation_cache_db_path())
        self.cache = TranslationCache(
            max_entries=config.model.cache_max_entries,
            ttl_seconds=config.model.cache_ttl_seconds,
            enabled=config.model.enable_cache,
            store=cache_store,
        )
        self.overlay = OverlayManager(config.style)

        # ---- 核心调度层 ----
        self.resource_monitor = ResourceMonitor()
        self.scheduler = ModelScheduler(config.schedule)
        self.coordinate = CoordinateMapper()

        # ---- 业务功能层 ----
        self.glossary: Glossary = self._load_glossary(config)
        self.preprocessor = TextPreprocessor.from_config(config)
        self.history = HistoryService(enabled=config.history.enabled)
        # 注意：向流水线与翻译服务传递的是 AppConfig（而非 SettingsManager）。
        # 它们只需要读取配置值；配置变更时由 _on_config_changed 推送新对象。
        self.translator = TranslationService(
            config=config,
            model_pool=self.model_pool,
            scheduler=self.scheduler,
            cache=self.cache,
            preprocessor=self.preprocessor,
            glossary=self.glossary,
            history=self.history,
        )

        # ---- 联网预翻译（D1：缓存直读常开，开关只管网络抓取/预热）----
        from ..services.prefetch import (
            NetworkMonitor,
            PrefetchGate,
            SentenceCache,
            SentenceCacheLookup,
        )
        from ..services.prefetch.version import fingerprint_for_config

        self.prefetch_cache = SentenceCache(
            path=paths.app_data_dir() / "prefetch_cache.db",
            model_version=fingerprint_for_config(config),
            max_bytes=config.prefetch.cache_max_mb,
        )
        self.prefetch_gate = PrefetchGate(
            enabled=config.prefetch.enabled and config.prefetch.consent_granted
            and not config.translation.local_only,
            online=False,
            health_failure_n=config.prefetch.health_failure_n,
            cooldown_s=config.prefetch.cooldown_s,
        )
        self.network_monitor = NetworkMonitor(
            probe_interval_s=config.prefetch.probe_interval_s,
            timeout_ms=config.prefetch.probe_timeout_ms,
            wifi_only=config.prefetch.wifi_only,
        )
        self.prefetch_lookup = SentenceCacheLookup(
            self.prefetch_cache,
            fuzzy_ratio=config.prefetch.fuzzy_ratio,
            fuzzy_window=config.prefetch.fuzzy_window_n,
            sequence_check=config.prefetch.fuzzy_sequence_check,
        )
        self.translator.set_prefetch_lookup(self.prefetch_lookup)
        self.network_monitor.state_changed.connect(self._on_network_state)
        if config.prefetch.enabled and not config.translation.local_only:
            self.network_monitor.start()

        # ---- 预翻译执行回路：活跃轨注册 → 调度器 → 本地模型 → 落库 ----
        from ..services.prefetch import (
            ActiveTrackRegistry,
            PrefetchScheduler,
        )
        from ..services.prefetch.econ_gate import EconGate
        from ..services.prefetch.metrics import PrefetchMetrics

        self.prefetch_metrics = PrefetchMetrics(
            path=paths.app_data_dir() / "prefetch_metrics.db",
            retention_days=config.prefetch.metrics_retention_days,
        )
        self.prefetch_econ_gate = EconGate(
            stats_provider=self.prefetch_metrics.site_stats,
            C=config.prefetch.econ_C_s,
            delta_L_s=config.prefetch.econ_delta_L_s,
            V=config.prefetch.econ_V,
            coldstart_per_site_day=config.prefetch.coldstart_probe_per_site_day,
        )
        self.track_registry = ActiveTrackRegistry(
            auto_advance=config.prefetch.auto_advance_track
        )
        self.prefetch_scheduler = PrefetchScheduler(
            idle_check=lambda: (
                self.resource_monitor.sample(force=False).cpu_percent
                < self.settings.config.prefetch.idle_cpu_threshold
                and self.model_pool.model_idle_seconds()
                >= self.settings.config.prefetch.idle_model_seconds
            ),
            gate_allow=lambda source: self.settings.config.prefetch.enabled
            and not self.settings.config.translation.local_only and (
                self.prefetch_gate.source_allowed(source)
                or any(track.track_id == source for track in self.track_registry.all_tracks())
            ),
            # 字幕预热不依赖联网（D5）：已注册字幕轨只要求开关开启。
            pull=self._pull_prefetch_job,
            model_queue_depth_max=config.prefetch.model_queue_depth_max,
            queue_depth_probe=lambda: self._ocr_queue_depth(),
        )
        self._wire_prefetch_progress()
        self._subtitle_watcher = None
        self._subtitle_watch_dirs: tuple[str, ...] = ()
        if config.prefetch.enabled and not config.translation.local_only:
            self.prefetch_scheduler.start()
            self._start_subtitle_watch(config)
        self.hotkeys = HotkeyService(on_action=self._on_hotkey)

        # ---- 流水线（跨核心调度层与 UI 层）----
        self.pipeline = Pipeline(
            config=config,
            capturer=self.capturer,
            differ=self.differ,
            ocr_engine=self.ocr_engine,
            translator=self.translator,
            overlay=self.overlay,
            resource_monitor=self.resource_monitor,
        )

        # 记录启动时的语言方向：首次配置变更才能与它比较，
        # 否则启动后的第一次语言切换不会触发整帧重识别
        self._last_translation_direction = (
            config.translation.source_language,
            config.translation.target_language,
        )
        self._last_api_settings = asdict(config.api)

    def _load_glossary(self, config: object) -> Glossary:
        """加载术语库。"""
        from ..services.glossary_service import create_glossary

        return create_glossary(config.translation)

    # ------------------------------------------------------------------
    # 装配：UI 交互层
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """装配界面组件。"""
        self.tooltips = TooltipController(self.theme)
        self.toasts = ToastManager(
            self.theme, enabled=self.settings.config.ui.notifications
        )

        self.tray = TrayIcon(self.theme)
        self.tray.show()

        self.panel = MainPanel(self)
        self.panel.setWindowIcon(self.app_icon())
        self.toasts.set_anchor(self.panel)

        self.floating_bar = FloatingBar(self.theme, self.tooltips)
        self.transcript_panel = TranscriptPanel(self.theme)
        self.region_selector = RegionSelector(self.theme)
        self.history_dialog: HistoryDialog | None = None

        self._refresh_ui_from_config()
        self._refresh_model_status()
        self._refresh_model_paths()
        self._refresh_runtime_info()
        self._refresh_hotkey_status()

    def _connect_signals(self) -> None:
        """连接全部信号。"""
        # 流水线
        self.pipeline.status_updated.connect(self._on_status_updated)
        self.pipeline.error_occurred.connect(self._on_pipeline_error)
        self.pipeline.capture_failed.connect(self._on_capture_failed)

        self.pipeline.skip_notice.connect(self._on_skip_notice)
        self.pipeline.transcript_updated.connect(self._on_transcript_updated)

        # 托盘
        self.tray.toggle_translation.connect(self.toggle_translation)
        self.tray.switch_model.connect(self.switch_model)
        self.tray.pick_region.connect(self.pick_region)
        self.tray.toggle_overlay.connect(self.toggle_overlay)
        self.tray.copy_result.connect(self.copy_result)
        self.tray.open_settings.connect(self.show_panel)
        self.tray.open_history.connect(self.open_history)
        self.tray.open_web_reader.connect(self.open_web_reader)
        self.tray.quit_requested.connect(self.quit)
        self.tray.activated.connect(self.show_panel)

        # 悬浮控制条
        self.floating_bar.toggle_translation.connect(self.toggle_translation)
        self.floating_bar.model_changed.connect(self.set_model)
        self.floating_bar.pick_region.connect(self.pick_region)
        self.floating_bar.open_settings.connect(self.show_panel)
        self.floating_bar.open_history.connect(self.open_history)
        self.floating_bar.position_changed.connect(self._on_bar_moved)
        self.floating_bar.toggle_transcript.connect(self.toggle_transcript_panel)
        self.transcript_panel.expanded_changed.connect(self._on_transcript_toggled)

        # 区域框选
        self.region_selector.region_selected.connect(self._on_region_selected)
        self.region_selector.cancelled.connect(self._on_region_cancelled)

        # 热键（跨线程 → 主线程）
        self.hotkey_triggered.connect(self._handle_hotkey_action)

        # 模型预加载（跨线程 → 主线程）
        self.preload_progress.connect(self._on_preload_progress)
        self.preload_finished.connect(self._on_preload_finished)

        # 后台维护完成（跨线程 → 主线程刷新界面）
        self.maintenance_finished.connect(self._on_maintenance_finished)

        # 未捕获异常报告（可能来自后台线程，用队列连接保证在主线程弹提示）
        self.exception_reported.connect(self._on_exception_reported)

        # 配置变更
        self.settings.changed.connect(self._on_config_changed)

        # 显示器变化
        app = QGuiApplication.instance()
        if app is not None:
            app.screenAdded.connect(lambda _screen: self._on_screens_changed())
            app.screenRemoved.connect(lambda _screen: self._on_screens_changed())
            app.primaryScreenChanged.connect(lambda _screen: self._on_screens_changed())

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """完成启动流程。"""
        config = self.settings.config
        reaped_count = self._pending_reaped_server_count
        self._pending_reaped_server_count = 0
        if reaped_count:
            self.notify(
                "已释放上次遗留的资源",
                f"清理了 {reaped_count} 个上次运行未退出的推理进程",
                "info", key="reap-stale-servers",
            )

        # 启动热键
        if config.hotkey.enabled:
            started = self.hotkeys.start(default_bindings(config.hotkey))
            # start() 内部会等注册完成，此时绑定结果才可用。
            # _build_ui 里的那次刷新发生在注册之前（bindings 为空），
            # 若没有这里补上的第二次刷新，"热键被占用"的面板提示与
            # 通知就永远不会出现——用户只会觉得"按了没反应"。
            # 全部注册失败时 start() 会把监听线程一并停掉，bindings()
            # 随之返回空表；不把返回值传下去，面板会把"全被占用"
            # 误展示成"未启用"，通知也永远发不出来。
            self._refresh_hotkey_status(started)

        # 检查关键依赖，给出可操作的提示而不是静默失败
        self._check_dependencies()

        # 界面呈现（先于模型加载：模型加载在后台线程进行，
        # 界面必须立即可用，否则用户会觉得程序启动很慢）
        if not config.ui.start_minimized:
            self.show_panel()
        if config.ui.floating_bar_visible:
            self._show_floating_bar()

        # 预加载模型
        self._start_preload()

        # 周期性维护任务
        self._maintenance_timer = QTimer(self)
        self._maintenance_timer.setInterval(_MAINTENANCE_INTERVAL_MS)
        self._maintenance_timer.timeout.connect(self._run_maintenance)
        self._maintenance_timer.start()

        logger.info("应用已启动")

    # ------------------------------------------------------------------
    # 模型预加载
    # ------------------------------------------------------------------

    def _start_preload(self) -> None:
        """在后台线程中预加载模型。

        延迟 300ms 再启动：让主窗口先完成首帧绘制。
        否则即使加载在后台线程，磁盘密集的读取仍会与界面初始化争抢 IO，
        导致首次显示出现明显延迟——用户感知到的"启动慢"就是这个。
        """
        config = self.settings.config
        # OCR is required for online translation too. Its first real frame
        # must not pay cold ONNX setup merely because local preload is disabled.
        self._warm_ocr_async()
        if not config.schedule.preload_on_start:
            logger.info("配置已关闭启动预加载")
            return

        targets = self.model_pool.preload_targets()
        if not targets:
            logger.info("没有需要预加载的模型（权重缺失或均已关闭预加载）")
            return

        QTimer.singleShot(300, self._begin_preload)

    def _reap_stale_servers(self) -> None:
        """清理上一次运行遗留的推理进程。

        程序被强制结束（任务管理器"结束任务"、崩溃、断电）时，清理代码没有
        机会执行，``llama-server`` 会变成孤儿继续占着内存和显存。实测攒到过
        8 个、合计 23.3 GB：内存占用 89%，显存被挤到使用共享内存，
        模型推理慢到超时——但 GPU 利用率显示 99%，看上去"在忙"，
        实际一个字都翻不出来。

        **必须在启动任何推理进程之前同步执行**：清理的判据是"可执行文件位于
        本程序运行时目录"，这同样适用于我们自己刚拉起的进程。放到后台线程里
        跑就可能误杀自己（psutil 全量扫描约百毫秒，同步执行可以接受）。

        只清理本程序运行时目录下的进程，不碰用户自己的 llama.cpp。
        """
        if getattr(self, "_stale_servers_reaped", False):
            return
        # Never repeat the sweep after a server owned by this instance could
        # have been created, even if this first attempt could not find a backend.
        self._stale_servers_reaped = True
        try:
            executable = self.model_pool.backend_probe().executable
        except Exception:  # noqa: BLE001 - 探测失败时跳过清理
            logger.debug("无法获取推理后端路径，跳过遗留进程清理", exc_info=True)
            return
        if not executable:
            return

        try:
            killed = reap_stale_servers(executable)
        except Exception:  # noqa: BLE001 - 清理失败不应影响启动
            logger.debug("清理遗留推理进程时出错", exc_info=True)
            return

        if killed:
            # UI has not been constructed yet. Present the existing notice
            # when start() runs instead of touching an uninitialized toast host.
            self._pending_reaped_server_count = len(killed)

    def _begin_preload(self) -> None:
        """真正发起预加载。"""
        if self.model_pool.is_preloading:
            logger.debug("预加载已在进行中，忽略重复请求")
            return

        # OCR 也要预热：RapidOCR 首次调用要加载 ONNX 模型（实测约 5.7 秒）。
        # 放在后台线程里做，避免首帧延迟。
        self._warm_ocr_async()

        labels = "、".join(
            MODEL_LABELS.get(model_id, model_id)
            for model_id in self.model_pool.preload_targets()
        )
        logger.info("开始预加载模型：%s", labels)

        started = self.model_pool.preload_async(
            # 回调运行在预加载线程中，通过信号切回主线程更新界面
            on_progress=lambda progress: self.preload_progress.emit(progress),
            on_finished=lambda progress: self.preload_finished.emit(progress),
        )
        if started:
            self.notify(
                "正在加载模型",
                labels,
                "info",
                key="model-preload",
            )

    def _warm_ocr_async(self) -> None:
        """在后台线程中预热 OCR 引擎。"""
        with self._ocr_warm_lock:
            if self._ocr_warm_running:
                return
            engine = self.ocr_engine
            generation = getattr(engine, "_backend_generation", 0)
            backend = getattr(engine, "_backend", None)
            if backend is not None and backend is self._ocr_warmed_backend:
                return
            self._ocr_warm_running = True

        def _worker() -> None:
            started = time.perf_counter()
            warmed_backend = None
            try:
                if engine.warm_up():
                    # Do not accidentally mark a newly selected cached backend
                    # as warmed when it replaces the one this worker warmed.
                    if generation == getattr(engine, "_backend_generation", 0):
                        warmed_backend = getattr(engine, "_backend", None)
                    logger.info("OCR 引擎预热完成（%.1fs）", time.perf_counter() - started)
                else:
                    logger.debug("OCR 引擎未预热（后端不可用或初始化失败）")
            except Exception:  # noqa: BLE001 - let a later request retry warm-up
                logger.debug("OCR 引擎预热失败，可再次预热", exc_info=True)
            finally:
                with self._ocr_warm_lock:
                    self._ocr_warmed_backend = warmed_backend
                    self._ocr_warm_running = False
                # Source changes can arrive while the old profile is warming.
                # Retry only for a changed generation; a failed warm-up alone
                # must not create a busy retry loop.
                if generation != getattr(engine, "_backend_generation", 0):
                    Application._warm_ocr_async(self)

        threading.Thread(target=_worker, name="OcrWarmUp", daemon=True).start()

    def _on_preload_progress(self, progress: object) -> None:
        """预加载进度更新（主线程）。

        托盘的提示文字随进度实时更新，但**通知只在开始加载一个新模型时发出**：
        同 key 的通知是"替换"语义（先关闭再新建），每次进度回调都发会导致
        通知肉眼可见地闪烁。
        """
        text = progress.as_text()
        self.tray.set_status_text(text)
        self.panel.set_preload_status(text, "info")

        if progress.completed == 0:
            self.notify("正在加载模型", text, "info", key="model-preload")

    def _on_preload_finished(self, progress: object) -> None:
        """预加载完成（主线程）。"""
        self._refresh_model_status()
        self._refresh_runtime_info()

        if progress.loaded:
            self.tray.set_status_text("")
            names = "、".join(
                MODEL_LABELS.get(model_id, model_id) for model_id in progress.loaded
            )
            self.panel.set_preload_status("模型已就绪", "success")
            self.notify(
                "模型已就绪",
                f"{names} 加载完成，可直接开始翻译",
                "success",
                key="model-preload",
            )
            logger.info("预加载完成：%s", names)
        else:
            # 全部失败：给出可操作的原因，而不是沉默
            reason = next(iter(progress.failed.values()), "模型不可用")
            self.tray.set_status_text(reason)
            self.panel.set_preload_status("模型未就绪", "warning")
            self.notify(
                "模型未加载",
                f"{reason}。当前为预览模式，译文将显示占位标记",
                "warning",
                key="model-preload",
            )
            logger.warning("预加载未成功：%s", progress.failed)

    def quit(self) -> None:
        """有序退出。"""
        logger.info("正在退出…")

        if self._maintenance_timer is not None:
            self._maintenance_timer.stop()

        self._prefetch_shutdown = True
        if getattr(self, "prefetch_scheduler", None) is not None:
            self.prefetch_scheduler.stop()
        if getattr(self, "network_monitor", None) is not None:
            self.network_monitor.stop()
        self._subtitle_watcher = None
        self.pipeline.shutdown()
        self.hotkeys.stop()
        self.model_pool.shutdown()
        self.cache.close()
        # 预取两个 SQLite 存储同样需要 checkpoint + 关句柄（UI 审查 F3）。
        if getattr(self, "prefetch_cache", None) is not None:
            self.prefetch_cache.close()
        if getattr(self, "prefetch_metrics", None) is not None:
            self.prefetch_metrics.close()
        self.history.close()
        self.settings.save()

        if self._single_instance is not None:
            self._single_instance.release()

        self.transcript_panel.hide_panel()
        self.tray.hide()
        self._qt_app.quit()
        logger.info("已退出")

    def set_single_instance(self, guard: SingleInstance) -> None:
        """注入单实例守卫。"""
        self._single_instance = guard

    # ------------------------------------------------------------------
    # 动作：翻译开关
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """实时翻译是否正在运行。"""
        return self.pipeline.is_running

    def toggle_translation(self) -> bool:
        """开启或暂停实时翻译。

        :return: 切换后是否处于运行状态。
        """
        if self.pipeline.is_running:
            self.pipeline.stop()
            self.overlay.clear()
            self._sync_running_state(False)
            self.notify("已暂停实时翻译", "屏幕内容不再被识别", "info")
            return False

        started = self.pipeline.start()
        if not started:
            # 启动失败的原因已在流水线内通过信号给出，这里不再重复提示
            self._sync_running_state(False)
            return False

        # 启动瞬间屏幕上已有内容需要被翻译，因此强制整帧识别
        self.pipeline.invalidate()
        self._sync_running_state(True)
        self.notify("已开始实时翻译", "正在识别并翻译屏幕内容", "success")
        return True

    def _sync_running_state(self, running: bool) -> None:
        """把运行状态同步到全部 UI 入口。"""
        self.tray.set_running(running)
        self.floating_bar.set_running(running)
        self.panel.set_running(running)

    # ------------------------------------------------------------------
    # 动作：模型
    # ------------------------------------------------------------------

    def switch_model(self) -> str:
        """在两个模型之间切换。"""
        model_id = self.scheduler.toggle_model()
        return self.set_model(model_id)

    def set_model(self, model_id: str) -> str:
        """切换到指定模型。"""
        self.scheduler.force_model(model_id)
        self.settings.update({"schedule": {"active_model": model_id}})

        self.tray.set_model(model_id)
        self.floating_bar.set_model(model_id)
        self.panel.set_model(model_id)

        label = MODEL_LABELS.get(model_id, model_id)
        available = self.model_pool.is_available(model_id)
        if available:
            # 用同一个 key：自动调度可能在短时间内多次切换模型，
            # 不用 key 的话通知会一条条堆到屏幕上限，把界面占满。
            self.notify("已切换模型", label, "success", key="model-switch")
        else:
            self.notify(
                "已切换模型",
                f"{label} · 权重未就绪，当前为预览模式",
                "warning",
                key="model-switch",
            )
        # 切换模型后缓存仍然有效（缓存键含语言与术语版本，不含模型），
        # 因此不清空缓存，避免重复推理相同文本。
        return model_id

    def refresh_models(self) -> None:
        """重新检测模型可用性并重新加载。

        用户下载权重后点"重新检测"时调用：先丢弃现有引擎实例
        （权重可能刚被放进去），再按新路径重建并触发加载。
        """
        self.model_pool.reset()
        self._refresh_model_status()
        self._refresh_model_paths()
        self._refresh_runtime_info()

        if self.model_pool.preload_targets():
            self._begin_preload()
        else:
            self.notify(
                "未找到模型权重",
                f"请将 GGUF 文件放入 {paths.project_models_dir()}，"
                f"或运行 scripts/download_models.py 自动下载",
                "warning",
                key="model-preload",
            )

    def clear_cache(self) -> None:
        """清空推理缓存。"""
        count = self.cache.clear()
        self._refresh_runtime_info()
        self.notify("缓存已清空", f"移除 {count} 条记录", "info")

    def open_models_dir(self) -> None:
        """在文件管理器中打开模型目录。"""
        from ..utils import paths

        self._open_path(str(paths.models_dir()))

    # ------------------------------------------------------------------
    # 动作：区域
    # ------------------------------------------------------------------

    def pick_region(self) -> None:
        """进入区域框选。"""
        if self.pipeline.is_running:
            # 框选期间暂停，避免刚选好的区域立即被翻译层覆盖
            self.pipeline.stop()
            self._sync_running_state(False)

        self.region_selector.start()

    def _on_region_selected(self, logical: Rect, monitor_index: int | None) -> None:
        """框选完成。"""
        self.settings.update(
            {
                "capture": {
                    "mode": "region",
                    "region": logical.to_sequence(),
                    # 用 is not None 而不是真值判断：mss 的显示器序号
                    # 从 1 起，当前 0 不会出现，但序号体系若变化，
                    # 0 号显示器会被这里静默丢弃。
                    "monitors": [monitor_index] if monitor_index is not None else [],
                }
            }
        )
        self.panel.set_region_text(f"{logical.width}×{logical.height} @({logical.x}, {logical.y})")
        self.pipeline.invalidate()
        self.notify("已设置翻译区域", f"{logical.width}×{logical.height}", "success")

    def _on_region_cancelled(self) -> None:
        """框选取消。"""
        self.notify("已取消区域选择", "", "info")

    def reset_region(self) -> None:
        """恢复为全屏翻译。"""
        self.settings.update({"capture": {"mode": "fullscreen", "region": None, "monitors": []}})
        self.panel.set_region_text("全屏")
        self.pipeline.invalidate()

    # ------------------------------------------------------------------
    # 动作：翻译层
    # ------------------------------------------------------------------

    def toggle_overlay(self) -> bool:
        """显示或隐藏翻译层。

        这是**低频**操作，因此允许淡入淡出；但译文层的**内容更新**
        不做任何动画（每秒刷新十余次，动画会有害）。
        """
        visible = not self.overlay.is_visible
        self.overlay.show_all(visible)
        self.settings.update({"ui": {"overlay_visible": visible}})

        self.tray.set_overlay_visible(visible)
        self.panel.set_overlay_visible_switch(visible)

        if visible:
            # 重新显示时需要重新识别，否则差分认为"没有变化"而不产出译文
            self.pipeline.invalidate()

        self.notify("翻译层已显示" if visible else "翻译层已隐藏", "", "info")
        return visible

    def set_overlay_visible(self, visible: bool) -> None:
        """设置翻译层可见性。"""
        if visible != self.overlay.is_visible:
            self.toggle_overlay()

    # ------------------------------------------------------------------
    # 动作：复制
    # ------------------------------------------------------------------

    def copy_result(self) -> None:
        """复制当前屏幕的翻译结果到剪贴板。"""
        from ..services.reading_order import reading_text

        items = [item for item in self.pipeline.snapshot_items() if item.is_renderable]
        text = reading_text(items)
        if not text.strip():
            self.notify("暂无可复制的译文", "请先开始翻译并等待识别结果", "warning")
            return

        clipboard = self._qt_app.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
            line_count = len(items)
            self.notify("已复制翻译结果", f"共 {line_count} 条", "success")

    # ------------------------------------------------------------------
    # 动作：面板与对话框
    # ------------------------------------------------------------------

    def show_panel(self) -> None:
        """显示主设置面板。"""
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

        self._refresh_runtime_info()
        self._refresh_model_status()

    def hide_panel(self) -> None:
        """隐藏主设置面板。"""
        self.panel.hide()

    def toggle_panel(self) -> None:
        """切换面板显示。"""
        if self.panel.isVisible():
            self.hide_panel()
        else:
            self.show_panel()

    def open_history(self) -> None:
        """打开翻译历史。"""
        if self.history_dialog is None:
            self.history_dialog = HistoryDialog(
                self.history, self.theme, self.toasts, self.panel
            )
        else:
            self.history_dialog.reload()

        self.history_dialog.show()
        self.history_dialog.raise_()
        self.history_dialog.activateWindow()

    # ------------------------------------------------------------------
    # 动作：术语库
    # ------------------------------------------------------------------

    def import_glossary(self) -> bool:
        """导入术语表 CSV。"""
        from pathlib import Path

        from PyQt6.QtWidgets import QFileDialog

        path_text, _ = QFileDialog.getOpenFileName(
            self.panel,
            "导入术语表",
            "",
            "CSV 文件 (*.csv);;所有文件 (*.*)",
        )
        if not path_text:
            return False

        from ..services.glossary_service import Glossary

        imported = Glossary(
            case_sensitive=self.settings.config.translation.glossary_case_sensitive
        )
        count = imported.load_from_csv(
            Path(path_text),
            case_sensitive=self.settings.config.translation.glossary_case_sensitive,
        )
        if count == 0:
            self.notify("导入失败", "未从文件中解析到有效术语，请检查格式", "error")
            return False

        # 记住路径，重启后自动加载
        self.settings.update(
            {"translation": {
                "glossary_path": path_text,
                "glossary_mode": "custom",
                "glossary_enabled": True,
            }}
        )
        self.glossary = imported
        self.translator.set_glossary(imported)
        self.panel.set_glossary_mode("custom")
        self.pipeline.invalidate()
        self._refresh_runtime_info()
        self.notify("术语表导入成功", f"共 {count} 条术语", "success")
        return True

    def select_glossary_mode(self, mode: str) -> bool:
        """Switch between the shipped terminology and the customer's CSV."""
        if mode not in {"builtin", "custom"}:
            return False
        if mode == "custom":
            from pathlib import Path

            path_text = self.settings.config.translation.glossary_path
            if not path_text or not Path(path_text).is_file():
                return self.import_glossary()
        self.settings.update({"translation": {"glossary_mode": mode}})
        self.reload_glossary()
        return True

    def select_glossary_domain(self, domain: str) -> bool:
        """Select a shipped subject pack without changing the user's CSV."""
        from ..glossary_catalog import BUILTIN_GLOSSARY_DOMAINS

        if domain not in BUILTIN_GLOSSARY_DOMAINS:
            return False
        self.settings.update({"translation": {"glossary_domain": domain}})
        if self.settings.config.translation.glossary_mode != "custom":
            self.reload_glossary()
        return True

    def export_glossary(self) -> None:
        """导出当前术语表。"""
        from pathlib import Path

        from PyQt6.QtWidgets import QFileDialog

        if len(self.glossary) == 0:
            self.notify("当前没有术语可导出", "", "warning")
            return

        path_text, _ = QFileDialog.getSaveFileName(
            self.panel, "导出术语表", "术语表.csv", "CSV 文件 (*.csv)"
        )
        if not path_text:
            return

        if self.glossary.save_to_csv(Path(path_text)):
            self.notify("术语表已导出", f"{len(self.glossary)} 条", "success")
        else:
            self.notify("导出失败", "请检查目标文件是否被占用", "error")

    def open_sample_glossary(self) -> None:
        """打开示例术语表文件。"""
        from ..utils import paths

        sample = paths.sample_glossary_path()
        if sample.exists():
            self._open_path(str(sample))
        else:
            self.notify("未找到示例文件", str(sample), "warning")

    def reload_glossary(self) -> None:
        """按当前配置重新加载术语库。"""
        from ..services.glossary_service import create_glossary

        self.glossary = create_glossary(self.settings.config.translation)
        self.translator.set_glossary(self.glossary)
        self.pipeline.invalidate()
        self._refresh_runtime_info()

    # ------------------------------------------------------------------
    # 动作：主题与动效
    # ------------------------------------------------------------------

    def set_reduced_motion(self, reduced: bool) -> None:
        """切换减少动态效果。"""
        apply_reduced_motion(reduced)
        self.notify(
            "已开启减少动态效果" if reduced else "已关闭减少动态效果",
            "界面过渡将更简短，去掉位移与缩放" if reduced else "",
            "info",
        )

    def _apply_reduced_motion_setting(self) -> None:
        """依据配置与系统设置初始化动效参数。

        设计规范要求：减少动态效果意味着"更少更柔和"，而不是完全取消——
        保留不透明度与颜色过渡，去掉位移与缩放。
        """
        config = self.settings.config
        reduced = bool(config.ui.reduced_motion)
        if not reduced and config.ui.respect_reduced_motion:
            reduced = _system_prefers_reduced_motion()
        apply_reduced_motion(reduced)
        if reduced:
            logger.info("已启用减少动态效果模式")

    def switch_theme(self, name: str | None = None) -> None:
        """切换深色 / 浅色主题。"""
        current = self.theme.name
        target = name or ("light" if current == "dark" else "dark")

        self.theme = build_theme(target)
        self.settings.update({"ui": {"theme": target}})

        self._qt_app.setStyleSheet(stylesheet(self.theme))
        self._apply_theme_to_widgets()
        self.notify("已切换主题", "浅色主题" if target == "light" else "深色主题", "info")

    def _apply_theme_to_widgets(self) -> None:
        """把新主题应用到全部界面组件。"""
        for widget in (
            self.panel,
            self.floating_bar,
            self.region_selector,
        ):
            if hasattr(widget, "set_theme"):
                widget.set_theme(self.theme)

        self.tray.set_theme(self.theme)
        self.toasts.set_theme(self.theme)
        self.tooltips.set_theme(self.theme)

    # ------------------------------------------------------------------
    # 动作：通知
    # ------------------------------------------------------------------

    def notify(
        self,
        title: str,
        description: str = "",
        kind: str = "info",
        key: str = "",
    ) -> None:
        """发送通知。

        :param kind: ``info`` / ``success`` / ``warning`` / ``error``。
        :param key: 去重标识；相同标识的通知会替换而不是叠加。
        """
        mapping = {
            "info": ToastKind.INFO,
            "success": ToastKind.SUCCESS,
            "warning": ToastKind.WARNING,
            "error": ToastKind.ERROR,
        }
        self.toasts.show(
            title=title,
            description=description,
            kind=mapping.get(kind, ToastKind.INFO),
            key=key,
        )

    def set_notifications(self, enabled: bool) -> None:
        """启用或禁用通知。"""
        self.toasts.set_enabled(enabled)

    def _on_exception_reported(self, title: str, description: str, kind: str) -> None:
        """在主线程展示未捕获异常的提示。

        槽必须留在主线程：异常可能来自后台线程，而 Qt 控件只能在主线程创建。
        通过信号连接（队列连接）转发，这里拿到的就是在主线程执行的调用。
        """
        self.notify(title, description, kind, key="uncaught-exception")

    # ------------------------------------------------------------------
    # 热键
    # ------------------------------------------------------------------

    def _on_hotkey(self, action: str) -> None:
        """热键回调。

        该方法运行在热键监听线程中，**不能直接操作 Qt 控件**。
        通过 pyqtSignal 转发到主线程执行（Qt 会自动使用队列连接）。
        """
        self.hotkey_triggered.emit(action)

    def _handle_hotkey_action(self, action: str) -> None:
        """在主线程中执行热键动作。"""
        if action == ACTION_TOGGLE_TRANSLATION:
            self.toggle_translation()
        elif action == ACTION_SWITCH_MODEL:
            self.switch_model()
        elif action == ACTION_COPY_RESULT:
            self.copy_result()
        elif action == ACTION_TOGGLE_OVERLAY:
            self.toggle_overlay()
        elif action == "show_panel":
            # 由单实例守卫在"程序被再次启动"时发出
            self.show_panel()
        else:
            logger.debug("未处理的热键动作：%s", action)

    # ------------------------------------------------------------------
    # 配置变更
    # ------------------------------------------------------------------

    def on_settings_changed(self, patch: dict) -> None:
        """由界面调用的配置变更入口。"""
        self._on_config_changed(self.settings.config)

    def _on_config_changed(self, config: object) -> None:
        """把配置变更应用到各组件。"""
        # 翻译语言方向：变更后必须强制整帧重识别，否则屏幕静止时差分
        # 会让流水线持续跳过，用户改了目标语言却看不到任何更新。
        # 只在方向真正变化时触发：样式等其它配置变更不值得白跑一次 OCR。
        direction = (config.translation.source_language, config.translation.target_language)
        direction_changed = (
            self._last_translation_direction is not None
            and direction != self._last_translation_direction
        )
        self._last_translation_direction = direction
        api_settings = asdict(config.api)
        api_changed = (
            self._last_api_settings is not None
            and api_settings != self._last_api_settings
        )
        self._last_api_settings = api_settings

        # 渲染样式
        self.overlay.apply_style(config.style)

        # OCR 参数
        ocr_generation = getattr(self.ocr_engine, "_backend_generation", 0)
        self.ocr_engine.configure(
            upscale_factor=config.ocr.upscale_factor,
            enable_preprocess=config.ocr.enable_preprocess,
            min_confidence=config.ocr.min_confidence,
            retain_uncertain=config.ocr.retain_uncertain,
            min_box_height=config.ocr.min_box_height,
            box_padding=config.ocr.box_padding,
            threads=config.ocr.threads,
            source_language=config.translation.source_language,
        )
        if ocr_generation != getattr(self.ocr_engine, "_backend_generation", 0):
            self._warm_ocr_async()
        self.pipeline.apply_config(config)

        # 预处理参数
        self.preprocessor.configure(config.preprocess)

        # 差分参数
        self.differ.configure(
            pixel_threshold=config.capture.diff_pixel_threshold,
            min_area_ratio=config.capture.diff_min_area_ratio,
            block_size=config.capture.diff_block_size,
        )

        # 调度
        self.scheduler.apply_config(config.schedule)
        self.model_pool.apply_config(
            config.model,
            idle_timeout=config.schedule.accurate_idle_timeout,
        )
        self.model_pool.apply_api_config(
            replace(config.api, enabled=False) if config.translation.local_only else config.api
        )

        # 缓存。启动时缓存关闭的应用实例没有持久层，用户在设置中开启后
        # 需要补接一次，否则本会话既不落盘也读不到历史条目。
        if config.model.enable_cache and self.cache.persistent_store is None:
            self.cache.attach_store(PersistentCacheStore(paths.translation_cache_db_path()))
        self.cache.configure(
            max_entries=config.model.cache_max_entries,
            ttl_seconds=config.model.cache_ttl_seconds,
            enabled=config.model.enable_cache,
        )

        # 翻译服务与历史
        self.translator.apply_config(config)
        self.history.enabled = bool(config.history.enabled)
        self._refresh_api_runtime_status()

        # 联网预翻译：开关只管网络抓取/预热；缓存直读常开（D1）
        prefetch = config.prefetch
        prefetch_scheduler = getattr(self, "prefetch_scheduler", None)
        self.prefetch_gate.set_enabled(
            prefetch.enabled and prefetch.consent_granted and not config.translation.local_only
        )
        if prefetch.enabled and not config.translation.local_only:
            self.network_monitor.start()
            if prefetch_scheduler is not None:
                prefetch_scheduler.start()
                self._start_subtitle_watch(config)
            self.panel.set_prefetch_badge(self.prefetch_gate.snapshot().badge)
        else:
            if prefetch_scheduler is not None:
                prefetch_scheduler.stop()
            self.network_monitor.stop()
            self._subtitle_watcher = None
            self._subtitle_watch_dirs = ()
            self.panel.set_prefetch_badge(self.prefetch_gate.snapshot().badge)

        if direction_changed:
            self.translator.clear_output()
            # The bundled catalog is materialized for the chosen target.
            self.reload_glossary()
            # Drop translations already queued or in flight under the old
            # direction; otherwise they can reappear after the user switches.
            self.pipeline.invalidate(clear_output=True)
            if self.transcript_panel.is_expanded:
                # 文本框展示的是"当前屏幕全部译文"的合并快照，
                # 语言换了以后旧快照的每一行都过期了，直接清空重等
                self.transcript_panel.set_text("（等待识别…）")
            logger.info(
                "翻译语言方向变更为 %s → %s，已请求整帧重新识别",
                direction[0],
                direction[1],
            )
        elif api_changed:
            self.translator.clear_output()
            # 静止画面的差分会跳过下一帧，且在途结果仍可能由旧服务返回。
            # 清空旧译文后强制重抓，使新 Key / 模型立即参与翻译。
            self.pipeline.invalidate(clear_output=True)
            if self.transcript_panel.is_expanded:
                self.transcript_panel.set_text("（等待识别…）")
            logger.info("在线 AI 配置已变更，已请求整帧重新翻译")

    def refresh_api_translation(self) -> None:
        """在线连接恢复后重新翻译静止画面，替换此前的本地回退结果。"""
        self.translator.clear_output()
        self.pipeline.invalidate(clear_output=True)
        if self.transcript_panel.is_expanded:
            self.transcript_panel.set_text("（等待识别…）")
        logger.info("在线 AI 连接恢复，已请求整帧重新翻译")

    # ------------------------------------------------------------------
    # 显示器变化
    # ------------------------------------------------------------------

    def _on_screens_changed(self) -> None:
        """显示器增删或几何变化后重新适配。"""
        self.coordinate.refresh()
        self.overlay.sync_screens()
        self.overlay.apply_style(self.settings.config.style)
        # 流水线自己也持有坐标映射（用于把框选区域换算成物理捕获区域），
        # 它的缓存同样必须刷新，否则区域模式仍按旧缩放比例捕获。
        self.pipeline.refresh_screens()

        # 重置框选覆盖层的几何，使其覆盖新的虚拟桌面
        self.region_selector._setup_window()
        self._push_ignore_regions()

        # 悬浮控制条此前可能停在已断开的显示器上：restore_position 的
        # 夹取只在显示时执行一次，正在显示的栏不会被自动拉回，用户会
        # 看到它"消失"（旧坐标仍持久化着）。屏幕变化后重新夹取。
        if self.floating_bar.isVisible():
            position = self.settings.config.ui.floating_bar_pos
            if position and len(position) == 2:
                self.floating_bar.restore_position(position[0], position[1])
            else:
                self.floating_bar.move_to_default_position()
            if self.transcript_panel.is_expanded:
                self.transcript_panel.expand(self.floating_bar)
            # 写回纠正后的位置，避免下次显示又从旧坐标恢复
            clamped = self.floating_bar.pos()
            self.settings.update(
                {"ui": {"floating_bar_pos": [clamped.x(), clamped.y()]}}
            )

        self.notify("显示器配置已变化", "翻译层已重新适配", "info")
        logger.info("显示器配置变化，已完成重新适配")

    # ------------------------------------------------------------------
    # 周期性维护
    # ------------------------------------------------------------------

    def _run_maintenance(self) -> None:
        """周期性维护的调度入口（在 Qt 主线程中被定时器调用）。

        **本方法只负责派发，不做任何实际工作。** 维护涉及 SQLite 查询、
        文件系统遍历、以及终止子进程（``wait()`` 最多阻塞 8 秒）——这些
        一旦在主线程执行，界面就会冻结。实测曾因 SQLite 锁的自死锁导致
        主线程永久卡住，程序显示"未响应"。

        因此实际工作放在后台线程，界面刷新通过信号切回主线程。
        """
        if self._maintenance_running:
            # 上一轮还没结束（例如磁盘很慢）就跳过本轮，避免线程堆积
            logger.debug("上一轮维护尚未结束，跳过本次")
            return

        self._maintenance_running = True
        threading.Thread(
            target=self._maintenance_worker, name="Maintenance", daemon=True
        ).start()

    def _maintenance_worker(self) -> None:
        """在后台线程中执行实际维护工作。

        这里**不得触碰任何 Qt 控件**，也不能发出需要主线程立即处理的信号
        （只发一个"完成"信号）。
        """
        try:
            self.model_pool.unload_idle()
            self.cache.purge_expired()

            max_entries = self.settings.config.history.max_entries
            if max_entries > 0:
                self.history.prune(max_entries)
        except Exception:  # noqa: BLE001 - 维护失败不应影响主流程
            logger.exception("维护任务执行失败")
        finally:
            self._maintenance_running = False
            # 关闭本线程的连接。历史服务用的是 threading.local 连接，
            # 本线程马上就要结束，不显式关闭会等到对象回收才释放文件句柄。
            self.history.close()

        # 界面刷新回主线程执行（信号跨线程会自动排队）
        self.maintenance_finished.emit()

    def _on_maintenance_finished(self) -> None:
        """维护完成后在主线程刷新界面信息。"""
        self._refresh_runtime_info()

    # ------------------------------------------------------------------
    # 状态刷新
    # ------------------------------------------------------------------

    def _on_status_updated(self, status: object) -> None:
        """流水线状态更新。"""
        self.floating_bar.set_status(
            f"≤{status.end_to_end_latency * 1000:.0f}ms"
            if getattr(status, "presentation_confirmed", False) else "等待显示"
        )
        self.panel.set_metrics(status.as_text())
        self._refresh_api_runtime_status()
        self.status_updated.emit(status)

        # 自动调度可能改变了模型，需要同步到界面
        model_id = self.scheduler.current_model
        if model_id != self._last_synced_model:
            self._last_synced_model = model_id
            self.tray.set_model(model_id)
            self.floating_bar.set_model(model_id)
            self.panel.set_model(model_id)

    def _on_network_state(self, probe) -> None:
        """探测结论 → 预翻译网关（OFFLINE/ONLINE 迁移打日志在状态机内），
        并把实际生效模式同步到设置页徽标——不允许开关与行为不一致。"""
        gate = self.prefetch_gate
        gate.set_online(probe.state.value == "online")
        self.panel.set_prefetch_badge(gate.snapshot().badge)

    def _ocr_queue_depth(self) -> int:
        """实时链路 OCR 队列深度（调度器让出条件的输入）。"""
        try:
            return len(self.pipeline._ocr_queue)
        except Exception:  # noqa: BLE001
            return 0

    def _pull_prefetch_job(self, job) -> bool:
        """执行一个预翻译任务：查行文本 → 本地模型翻译 → 写入缓存。

        字幕任务（kind=subtitle_line）：payload 为行号；
        网页任务（P3）payload 为 URL。返回 True 表示有产出。
        """

        if self._stop_prefetch_requested:
            return False
        if job.kind == "web_page":
            return self._prefetch_web_page(str(job.payload))
        if job.kind != "subtitle_line":
            return False
        track_id = job.track_or_site
        with self._subtitle_lines_lock:
            line = self._subtitle_lines.get(track_id, {}).get(int(job.payload))
        if line is None:
            return False

        config = deepcopy(self.settings.config)
        revision = self.translator._request_revision
        namespace = self.translator._live_namespace()
        target = config.translation.target_language
        translated = self.translator.translate_text(line["source_raw"])
        if (
            not translated or config != self.settings.config
            or revision != self.translator._request_revision
            or self._stop_prefetch_requested
            or not self.translator._valid_output(line["source_raw"], translated)
        ):
            return False
        with self._subtitle_lines_lock:
            if self._subtitle_lines.get(track_id, {}).get(int(job.payload)) is not line:
                return False
        if (
            namespace.startswith("api:") or namespace != self.translator._live_namespace()
            or config.translation.concise_output
        ):
            # Provider/concise identities belong to the verified inference
            # cache; the shared subtitle schema cannot represent those scopes.
            return True

        from ..services.language import detect_language
        from ..services.prefetch.sentence_cache import SentenceEntry

        source = config.translation.source_language
        if source == "auto":
            source = detect_language(
                line["source_raw"], config.translation.fallback_source_language
            ).code
        entry = SentenceEntry(
            source=line["source_raw"],
            translated=translated,
            src_lang=source,
            tgt_lang=target,
            domain=track_id,
            model_version=self.prefetch_cache._model_version,
            source_kind="subtitle",
            ttl_days=config.prefetch.subtitle_ttl_days,
            line_no=int(job.payload),
            start_ms=line["start_ms"],
            end_ms=line["end_ms"],
        )
        # A reimport can happen while the model works. Hold the registration
        # lock through the final identity check and write so an old in-flight
        # line cannot replace the new track's index after it was registered.
        with self._subtitle_lines_lock:
            if self._subtitle_lines.get(track_id, {}).get(int(job.payload)) is not line:
                return False
            self.prefetch_cache.put_many([entry])
        return True

    def _wire_prefetch_progress(self) -> None:
        self.prefetch_progress.connect(
            self._on_prefetch_progress, Qt.ConnectionType.QueuedConnection
        )
        self.prefetch_scheduler.progress = self.prefetch_progress.emit

    def _start_subtitle_watch(self, config) -> None:
        """启动字幕目录监视（仅联网预翻译开启时）。"""
        from ..services.prefetch.active_track import SubtitleTrack
        from ..services.prefetch.normalizer import normalize_for_key
        from ..services.prefetch.subtitle_source import (
            SubtitleDirectoryWatcher,
            parse_subtitle_file,
        )

        directories = tuple(config.prefetch.subtitle_dirs)
        if not directories:
            # 用户清空字幕目录：必须丢弃旧监视器，否则旧目录里新放入的
            # 字幕文件仍会被解析预翻译（UI 审查 F1，major）。
            self._subtitle_watcher = None
            self._subtitle_watch_dirs = ()
            return
        if (
            self._subtitle_watcher is not None and directories == self._subtitle_watch_dirs
        ):
            return

        def on_file(path) -> None:
            try:
                lines = parse_subtitle_file(path, normalize_for_key)
            except Exception as exc:
                logger.warning("字幕解析失败 %s：%s", path, exc)
                self.notify("字幕解析失败", f"{path.name}：{exc}", "warning")
                return
            if not lines:
                return
            track = SubtitleTrack(
                track_id=str(path),
                path=str(path),
                title=path.name,
                line_count=len(lines),
            )
            self.register_subtitle_track(track, lines)
            self.notify(
                "已注册字幕轨",
                f"{path.name}（{len(lines)} 行）开始预翻译",
                "success",
            )

        self._subtitle_watcher = SubtitleDirectoryWatcher(
            config.prefetch.subtitle_dirs, on_file
        )
        self._subtitle_watch_dirs = directories

    def _on_prefetch_progress(self, page_key: str, done: int, total: int) -> None:
        """预热进度：面板文案 + 完成通知。"""
        if total <= 0:
            return
        failed = self.prefetch_scheduler.page_failures(page_key)
        track = None
        if page_key.startswith("track:"):
            track_id = page_key[len("track:"):]
            for candidate in self.track_registry.all_tracks():
                if candidate.track_id == track_id:
                    track = candidate
                    break

        if done >= total:
            name = track.title if track is not None else page_key
            logger.info("字幕预翻译完成：%s（%s 行）", name, total)
            self.notify("字幕预翻译完成", f"{name}（{total} 行）已全部就绪", "success")
            self.panel.set_prefetch_progress("")
            return

        percent = round(done / total * 100)
        if track is not None:
            failure_text = f"，{failed} 行失败" if failed else ""
            self.panel.set_prefetch_progress(
                f"《{track.title}》预翻译 {percent}%（{done}/{total} 行{failure_text}）"
            )

    def _prefetch_web_page(self, url: str) -> bool:
        """后台预取一个网页：抓取 → 抽取 → 逐块翻译 → 落库（P3）。"""
        from ..services.language import detect_language
        from ..services.prefetch.sentence_cache import SentenceEntry
        from ..services.prefetch.web_fetcher import extract_article, fetch

        if self.prefetch_gate.snapshot().gate.value != "预翻译中":
            return False
        result = fetch(
            url, timeout_s=15.0,
            max_bytes=self.settings.config.prefetch.max_page_bytes,
        )
        if result.kind != "ok":
            self.prefetch_gate.notify_fetch_failed(
                url.split("/")[2] if "//" in url else url, result.reason or result.kind
            )
            return False
        article = extract_article(result.html, result.final_url, result.etag)
        if not article.text_blocks:
            return False

        host = url.split("/")[2] if "//" in url else url
        config = deepcopy(self.settings.config)
        revision = self.translator._request_revision
        namespace = self.translator._live_namespace()
        target = config.translation.target_language
        written = 0
        for block in article.text_blocks:
            if self._stop_prefetch_requested:
                break
            translated = self.translator.translate_text(block)
            if (
                config != self.settings.config or self._stop_prefetch_requested
                or revision != self.translator._request_revision
            ):
                break
            if not translated or not self.translator._valid_output(block, translated):
                continue
            if (
                namespace.startswith("api:") or namespace != self.translator._live_namespace()
                or config.translation.concise_output
            ):
                # translate_text already warmed the scoped inference cache.
                written += 1
                continue
            source = config.translation.source_language
            if source == "auto":
                source = detect_language(block, config.translation.fallback_source_language).code
            self.prefetch_cache.put_many([SentenceEntry(
                source=block, translated=translated, src_lang=source,
                tgt_lang=target, domain=host,
                model_version=self.prefetch_cache._model_version,
                source_kind="web", ttl_days=config.prefetch.site_ttl_days,
                etag=article.etag, bytes_in=len(block.encode("utf-8")),
            )])
            self.prefetch_metrics.on_prefetch(
                host + ":" + str(written), host, len(block.encode("utf-8"))
            )
            written += 1
        return written > 0

    @property
    def _stop_prefetch_requested(self) -> bool:
        """预取中途的停止开关：用户关闭联网预翻译时立即中止当前页面，
        不再把剩余文本块逐段送模型。阅读器窗口生命周期信号接入后在此
        一并纳入。"""
        try:
            return (not self.settings.config.prefetch.enabled
                    or self.settings.config.translation.local_only
                    or getattr(self, "_prefetch_shutdown", False))
        except Exception:  # noqa: BLE001 - 配置暂不可读时保守继续
            return False

    def register_subtitle_track(self, track, lines=None) -> None:
        """注册字幕轨并入队预翻译（P2 尾：导入/目录监视调用）。"""
        from ..services.prefetch.scheduler import PrefetchJob, Priority

        if lines is None:
            from ..services.prefetch.normalizer import normalize_for_key
            from ..services.prefetch.subtitle_source import parse_subtitle_file

            lines = parse_subtitle_file(track.path, normalize_for_key)
        # Raw input must exist before jobs are visible to the worker. Reading
        # it from the translated cache cannot bootstrap a newly imported file.
        snapshot = {
            line.line_no: {
                "source_raw": line.text_raw, "start_ms": line.start_ms, "end_ms": line.end_ms,
            } for line in lines
        }
        with self._subtitle_lines_lock:
            self._subtitle_lines[track.track_id] = snapshot
        self.track_registry.register(track)
        if self.prefetch_lookup._active_track_id != self.track_registry.active_id:
            self.prefetch_lookup.set_active_track(self.track_registry.active_id)
        page_key = f"track:{track.track_id}"
        self.prefetch_scheduler.cancel_page(page_key)
        self.prefetch_scheduler.clear_cancelled(page_key)
        if not self.settings.config.prefetch.enabled:
            return
        for line_no in sorted(snapshot):
            self.prefetch_scheduler.enqueue(PrefetchJob(
                priority=Priority.PLAYBACK_CURRENT, seq=0,
                kind="subtitle_line", track_or_site=track.track_id,
                payload=line_no, page_key=page_key,
            ))
        self.prefetch_scheduler.start()

    def open_web_reader(self, url: str = "") -> None:
        """打开网页对照阅读器（P3；D2 体验边界：浏览器内仍走屏幕翻译）。"""
        from .web_reader_dialog import WebReaderDialog

        if getattr(self, "_web_reader", None) is None or not self._web_reader.isVisible():
            self._web_reader = WebReaderDialog(self)
        self._web_reader.open_url(url)

    def clear_prefetch_cache(self) -> None:
        """用户点"清除缓存"。"""
        self.prefetch_cache.clear()
        logger.info("预翻译缓存已清空")

    def prefetch_cache_usage_text(self) -> str:
        """缓存占用的人读文本。"""
        usage = self.prefetch_cache.usage_bytes()
        if usage >= 1024 * 1024:
            return f"缓存占用 {usage / 1024 / 1024:.1f} MB"
        return f"缓存占用 {usage / 1024:.0f} KB"

    def _on_capture_failed(self) -> None:
        """捕获线程连续失败退出：整条停机，状态位与实际一致。"""
        self.pipeline.stop()

    def _on_pipeline_error(self, message: str) -> None:
        """流水线错误：同步状态并提示用户可操作的原因。"""
        logger.error("流水线错误：%s", message)
        self._sync_running_state(False)
        self.tray.set_status_text(message)
        self.notify("实时翻译无法启动", message, "error", key="pipeline-error")

    def _on_skip_notice(self, reason: str, target_language: str) -> None:
        """整帧内容没有可见变化时，明确告诉用户原因与下一步怎么做。

        这是"点了开始翻译却什么都没变"最常见的成因：屏幕文字本身
        已经是目标语言。程序此时按设计原样输出，外观与"没反应"完全
        一样，因此必须由界面把原因讲清楚——否则用户只会认为软件坏了。
        """
        if reason == "region_offscreen":
            logger.warning("已提示用户：框选区域已离开屏幕 %s", target_language)
            self.tray.set_status_text("框选区域已离开屏幕，已回退整屏翻译")
            self.notify(
                "框选区域已离开屏幕",
                "框选的翻译区域所在的显示器可能已被拔掉或关闭。"
                "程序已临时改为整屏捕获，翻译仍可继续；"
                "若不需要整屏翻译，请重新框选区域。",
                "warning",
                key="region-offscreen",
            )
            return
        if reason != "already_target":
            return

        from ..languages import LANGUAGES

        info = LANGUAGES.get(target_language)
        name = info.name if info is not None else target_language
        logger.info("已提示用户：屏幕内容已是目标语言「%s」", name)
        self.tray.set_status_text(f"屏幕文字已是{name}，无需翻译")
        self.notify(
            "没有需要翻译的内容",
            f"当前目标语言是{name}，而屏幕上识别到的文字已经是{name}，"
            f"因此没有可翻译的内容。若要翻译外语界面，请在设置中把目标语言"
            f"改成{name}之外的语言。",
            "info",
            key="skip-same-language",
        )

    # ------------------------------------------------------------------
    # 实时译文文本框
    # ------------------------------------------------------------------

    def toggle_transcript_panel(self) -> bool:
        """展开或收起实时译文文本框。"""
        expanded = self.transcript_panel.toggle(self.floating_bar)
        return expanded

    def _push_ignore_regions(self) -> None:
        """Keep app controls free of OCR work and underlying overlay text."""
        from ..components.capture.geometry import Rect

        rects: list[Rect] = []

        def to_physical(widget) -> Rect | None:
            # Include native title bars and borders as well as the client area.
            frame = widget.frameGeometry()
            logical = Rect(frame.x(), frame.y(), frame.width(), frame.height())
            try:
                return self.coordinate.logical_to_physical(logical)
            except Exception:  # noqa: BLE001 - 几何异常不应影响翻译
                return None

        if self.floating_bar.isVisible():
            rect = to_physical(self.floating_bar)
            if rect is not None:
                rects.append(rect)
        # 文本框用"目标高度"矩形：展开动画期间 height() 还在从 0 长大，
        # 按当前高度推送会让动画长出来的部分脱离忽略区，文字被 OCR 读走
        panel_rect = self.transcript_panel.ignorable_rect
        if panel_rect is not None:
            try:
                physical_rect = self.coordinate.logical_to_physical(panel_rect)
                rects.append(physical_rect)
            except Exception:  # noqa: BLE001 - 几何异常不应影响翻译
                pass

        qt_app = getattr(self, "_qt_app", None)
        if qt_app is not None:
            from ..components.render.overlay_window import OverlayWindow

            for widget in qt_app.topLevelWidgets():
                if (
                    not widget.isVisible() or isinstance(widget, OverlayWindow)
                    or not type(widget).__module__.startswith("pingyitong.")
                ):
                    continue
                rect = to_physical(widget)
                if rect is not None:
                    rects.append(rect)

        # Stable geometry avoids resetting the OCR baseline on repeated events.
        rects = sorted(set(rects), key=lambda rect: rect.to_ltrb())
        changed = rects != getattr(self, "_last_owned_window_regions", None)
        self._last_owned_window_regions = rects
        self.pipeline.set_ignore_regions(rects)
        if hasattr(self.overlay, "set_excluded_regions"):
            self.overlay.set_excluded_regions(rects)
        if changed:
            # Closing or moving an excluded window exposes unchanged pixels.
            # They need one fresh capture even though the desktop diff is zero.
            self.pipeline.invalidate()

    def _on_owned_windows_changed(self) -> None:
        self._push_ignore_regions()

    def _on_transcript_toggled(self, expanded: bool) -> None:
        """文本框展开状态变化时切换显示方式。

        展开文本框意味着"改用文本方式阅读"：此时隐藏屏幕上的原位译文层，
        否则同一段文字会在屏幕上和文本框里各出现一次，既乱又费性能。
        关闭时恢复用户原本的翻译层显示设置。

        文本框的展开状态**不写入配置**：它是一次性的阅读方式切换，
        不是长期偏好；下次启动仍按用户设置的翻译层状态呈现。
        """
        self.floating_bar.set_transcript_open(expanded)
        self.panel.set_transcript_open(expanded)

        if expanded:
            self._overlay_visible_before_transcript = self.overlay.is_visible
            self.overlay.show_all(False)
            self.pipeline.set_transcript_enabled(True)
            if not self.transcript_panel.text().strip():
                self.transcript_panel.set_text("（等待识别…）")

            # 文本框的意义就是实时显示译文；流水线还停着时它只会永远
            # 停在"等待识别"，用户只能认为功能坏了。展开即自动开始翻译
            # （收起不自动暂停——是否停止由用户决定）。
            if not self.pipeline.is_running:
                logger.info("文本框模式：流水线未运行，自动开始实时翻译")
                self.toggle_translation()

            self._push_ignore_regions()
            logger.info("已展开实时译文文本框，原位翻译层暂时隐藏")
        else:
            self.pipeline.set_transcript_enabled(False)
            restore = getattr(self, "_overlay_visible_before_transcript", None)
            if restore is None:
                restore = bool(self.settings.config.ui.overlay_visible)
            self.overlay.show_all(bool(restore))
            self.pipeline.invalidate()
            self._push_ignore_regions()
            logger.info("已收起实时译文文本框，恢复原位翻译层")

    def _on_transcript_updated(self, items: object) -> None:
        """把最新的译文快照按阅读顺序写入文本框。"""
        from ..services.reading_order import reading_text

        text = reading_text(list(items or []))
        self.transcript_panel.set_text(text if text else "（当前屏幕没有可翻译的文字）")

    def _refresh_ui_from_config(self) -> None:
        """按配置初始化界面状态。"""
        config = self.settings.config

        self.tray.set_overlay_visible(config.ui.overlay_visible)
        self.tray.set_model(config.schedule.active_model)
        self.floating_bar.set_model(config.schedule.active_model)
        self.panel.set_model(config.schedule.active_model)

        # 预翻译徽标 = 状态机实际结论（不是开关勾选态）
        if config.prefetch.enabled:
            self.panel.set_prefetch_badge(self.prefetch_gate.snapshot().badge)
        self.panel.set_prefetch_usage(self.prefetch_cache_usage_text())
        self._last_synced_model = config.schedule.active_model

        if config.capture.mode == "region" and config.capture.region:
            region = Rect.from_sequence(config.capture.region)
            self.panel.set_region_text(
                f"{region.width}×{region.height} @({region.x}, {region.y})"
            )
        else:
            self.panel.set_region_text("全屏")

        self.overlay.show_all(config.ui.overlay_visible)

    def _refresh_hotkey_status(self, started: bool | None = None) -> None:
        """把热键注册结果显示到设置面板。

        热键被占用时"按了没反应"对用户完全不可诊断，而这是常见情况
        （截图工具、输入法、录屏软件都会抢 Ctrl+Alt+字母 这类组合）。
        因此把每个热键的成功/失败状态直接写进面板。

        :param started: ``hotkeys.start()`` 的返回值。``None`` 表示界面
            初次构建时还没尝试注册（此时沿用"未启用"占位文案）；
            ``False`` 表示尝试过但一个热键都没注册上，必须明确告知用户
            而不是静默——部分失败与全部失败的体验要一致。
        """
        bindings = self.hotkeys.bindings()
        # start() 全失败时会 stop() 掉监听线程，bindings() 随之返回空表；
        # 只看 bindings 无法区分"没启用"和"全被占用"，必须结合返回值。
        all_failed = started is False or (bool(started) and not bindings)
        self.tray.set_hotkey_status(bindings)
        self.panel.set_hotkey_status(bindings, all_failed=all_failed)

        if all_failed:
            logger.warning("热键全部注册失败（可能被其他程序占用）")
            self.notify(
                "热键全部注册失败",
                "所有全局热键都未能注册（可能被其他程序占用），按热键不会有任何反应"
                " —— 可在设置面板查看并改用其他组合键",
                "warning",
                key="hotkey-all-failed",
            )
            return

        fallbacks = [b for b in bindings if b.registered and b.fallback_from]
        if fallbacks:
            from ..services.hotkey_service import format_sequence

            detail = "、".join(
                f"{format_sequence(b.fallback_from)} → {format_sequence(b.sequence)}"
                for b in fallbacks
            )
            self.notify(
                "快捷键已自动调整",
                f"原组合被占用，已改用 {detail}；可在设置页查看实际热键",
                "warning", key="hotkey-fallback",
            )

        conflicts = [b for b in bindings if not b.registered]
        if conflicts:
            from ..services.hotkey_service import ACTION_LABELS, format_sequence

            detail = "、".join(
                f"{format_sequence(b.sequence)}（{ACTION_LABELS.get(b.action, b.action)}）"
                for b in conflicts
            )
            logger.warning("以下热键被占用，无法响应：%s", detail)
            self.notify(
                "部分热键被其他软件占用",
                f"{detail} —— 可在设置面板查看并改用其他组合键",
                "warning",
                key="hotkey-conflict",
            )

    def _refresh_model_status(self) -> None:
        """刷新模型可用状态显示。"""
        for model_id in (MODEL_FAST, MODEL_ACCURATE):
            status = self.model_pool.engine(model_id).status()
            if status.available:
                tone = "success"
                text = "已就绪" if not status.loaded else "已加载"
            else:
                tone = "warning"
                text = "未就绪"

            self.panel.set_model_status(model_id, text, tone)

    def _refresh_model_paths(self) -> None:
        """刷新模型文件的解析路径显示。

        把实际查找结果展示出来，用户就能自己判断"为什么没加载到"——
        是文件没放对位置，还是文件名不匹配。
        """
        resolved: dict[str, str] = {}
        for model_id in (MODEL_FAST, MODEL_ACCURATE):
            file_name = self.settings.config.model.for_model(model_id).file_name
            found = paths.find_model_file(file_name)
            resolved[model_id] = str(found) if found else "未找到"

        self.panel.set_model_paths(resolved, str(paths.project_models_dir()))

    def _refresh_runtime_info(self) -> None:
        """刷新缓存、术语库与历史数据库的统计信息。"""
        self.panel.set_cache_stats(
            f"{len(self.cache)} 条 · {self.cache.stats().as_text()}"
        )
        self.panel.set_glossary_stats(self.glossary.stats_text())
        self.panel.set_history_size(f"{self.history.database_size_mb():.1f} MB")
        self._refresh_api_runtime_status()

    def _refresh_api_runtime_status(self) -> None:
        """把实际在线请求/回退状态同步到现有徽标。"""
        api = self.settings.config.api
        status = self.model_pool.api_status()
        if not api.enabled:
            text, tone = "未启用", "info"
        elif status is None or not status.available:
            reason = status.message if status is not None else "配置不完整"
            text, tone = f"本地回退 · {reason}", "warning"
        elif not status.total_calls:
            text, tone = f"已启用 · 待请求 · {api.model_name}", "info"
        elif not status.last_latency:
            text, tone = "上次请求失败 · 本地回退", "warning"
        else:
            text, tone = f"在线 AI · {api.model_name}", "success"
        self.panel.set_api_status(text, tone)

    def _show_floating_bar(self) -> None:
        """显示悬浮控制条并恢复位置。"""
        config = self.settings.config
        position = config.ui.floating_bar_pos
        if position and len(position) == 2:
            self.floating_bar.restore_position(position[0], position[1])
        else:
            self.floating_bar.move_to_default_position()
        self.floating_bar.show_bar(animate=False)
        self._push_ignore_regions()

    def _on_bar_moved(self, x: int, y: int) -> None:
        """持久化悬浮条位置，并让文本框跟随移动。"""
        self.settings.update({"ui": {"floating_bar_pos": [x, y]}})
        # 文本框贴着悬浮栏，栏被拖走后它若留在原地会显得与操作无关
        if self.transcript_panel.is_expanded:
            self.transcript_panel.expand(self.floating_bar)
        self._push_ignore_regions()

    # ------------------------------------------------------------------
    # 依赖检查
    # ------------------------------------------------------------------

    def _check_dependencies(self) -> None:
        """检查关键依赖并把结果告知用户。

        对全离线工具而言，"为什么不能翻译"必须能被用户自己诊断——
        静默失败会让用户以为软件坏了。
        """
        problems: list[str] = []

        if not ScreenCapturer.is_available():
            problems.append("未安装 python-mss，无法捕获屏幕")
        preflight = getattr(self.ocr_engine, "preflight_error", None)
        ocr_error = preflight() if callable(preflight) else (
            "未安装 RapidOCR，无法识别屏幕文字" if not self.ocr_engine.is_ready() else ""
        )
        if ocr_error:
            problems.append(str(ocr_error))
        if not self.model_pool.any_available():
            problems.append(
                "未找到可用的 Spark-X2.5 模型权重，当前为预览模式"
                "（译文位置可正常显示，但内容为占位标记）"
            )

        if not problems:
            logger.info("依赖检查通过")
            return

        for problem in problems:
            logger.warning("依赖检查：%s", problem)

        self.tray.set_status_text(problems[0])
        self.notify(
            "部分组件未就绪",
            problems[0],
            "warning",
            key="dependency-warning",
        )

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def app_icon(self) -> QIcon:
        """返回应用图标。

        图标来源为 PPT 封面展示的软件图标（``assets/icons/app_icon.*``）。
        """
        for file_name in (ICON_APP_ICO, ICON_APP_PNG):
            path = icon_path(file_name)
            if path.exists():
                icon = QIcon(str(path))
                if not icon.isNull():
                    return icon
        return QIcon()

    def _open_path(self, path: str) -> None:
        """用系统默认程序打开文件或目录。"""
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            self.notify("无法打开", path, "warning")


# ----------------------------------------------------------------------
# 系统设置探测
# ----------------------------------------------------------------------


def _system_prefers_reduced_motion() -> bool:
    """检测系统是否开启了"减少动画"。

    Windows 上对应"辅助功能 → 视觉效果 → 动画效果"开关，
    通过 ``SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION)`` 读取。
    非 Windows 平台或调用失败时返回 ``False``（即不启用减少动效）。
    """
    if sys.platform != "win32":
        return False

    try:
        spi_getclientareaanimation = 0x1042
        enabled = ctypes.c_int(1)
        ok = ctypes.windll.user32.SystemParametersInfoW(
            spi_getclientareaanimation, 0, ctypes.byref(enabled), 0
        )
        if not ok:
            return False
        return not bool(enabled.value)
    except Exception:  # noqa: BLE001 - 探测失败时按"不需要减少"处理
        return False


def _set_windows_app_id() -> None:
    """设置 Windows 任务栏标识。

    必须在创建任何窗口之前调用，否则任务栏会把程序归到 python.exe 下，
    导致图标显示错误、且无法固定到任务栏。
    """
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:  # noqa: BLE001
        logger.debug("设置任务栏标识失败（不影响功能）")


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------


def run(argv: list[str] | None = None, portable: bool | None = None) -> int:
    """程序主入口。

    :param argv: 命令行参数；``None`` 时使用 ``sys.argv``。
    :param portable: 便携模式；``None`` 时从 ``argv`` 里识别 ``--portable``。
        :mod:`pingyitong.__main__` 在预处理参数时已把 ``--portable`` 移除，
        因此必须显式传入，否则携带其他参数启动时便携模式会静默失效。
    :return: 进程退出码。
    """
    argv = list(argv if argv is not None else sys.argv)
    if portable is None:
        portable = "--portable" in argv

    from ..utils import paths

    paths.set_portable(portable)
    setup_logging()

    # 全局异常钩子必须尽早安装：PyQt6 遇到事件处理器里的未捕获异常会直接
    # qFatal() 终止进程（不打印堆栈），装钩子后才会交给我们处理。
    install_exception_hooks()

    _set_windows_app_id()

    # 单实例：重复启动时唤醒已有窗口而不是开新进程
    guard = SingleInstance(APP_NAME_EN)
    if not guard.acquire():
        guard.notify_existing()
        logger.info("检测到已有实例在运行，已唤醒该实例并退出")
        return 0

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName(APP_NAME)
    qt_app.setApplicationDisplayName(APP_NAME)
    qt_app.setOrganizationName(APP_ORG)
    qt_app.setApplicationVersion(version_string())
    # 主面板关闭时程序不退出——它是托盘常驻工具
    qt_app.setQuitOnLastWindowClosed(False)

    application = Application(qt_app, portable=portable)
    application.set_single_instance(guard)

    # 把异常提示接到界面上：异常可能在任意线程抛出，因此走信号转发
    # （Qt 会自动使用队列连接），避免在后台线程创建控件。
    install_exception_hooks(application.exception_reported.emit)

    qt_app.setWindowIcon(application.app_icon())
    qt_app.setStyleSheet(stylesheet(application.theme))

    # 已有实例被再次启动时，把主面板带到前台
    guard.start_listener(lambda: application.hotkey_triggered.emit("show_panel"))

    if not QSystemTrayIcon.isSystemTrayAvailable():
        logger.warning("系统不支持托盘图标，程序将以窗口模式运行")

    application.start()
    # 非托盘退出路径（如 Windows 注销/关机触发 WM_ENDSESSION）不会走
    # Application.quit()，这里挂兜底清理，避免 llama-server 成为孤儿、
    # 本次会话配置改动不落盘（UI 审查 F2，major）。quit 内部的
    # _qt_app.quit() 重入无害。
    qt_app.aboutToQuit.connect(application.quit)

    # 处理"不支持托盘"的极端情况：至少显示主面板，否则程序无入口
    if not QSystemTrayIcon.isSystemTrayAvailable():
        application.show_panel()

    return qt_app.exec()


