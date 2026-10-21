"""异步流水线调度（自研）。

对应《项目原创性声明》二·（一）"端到端实时桌面翻译全链路架构"。

流水线结构::

    捕获线程        捕获桌面画面 ──┐
                                  │  LatestQueue（丢旧保新）
    OCR 线程        文字识别 ─────┤
                                  │  LatestQueue
    翻译线程        预处理+推理 ──┤
                                  │  LatestQueue（仅保留最新一帧）
    主线程渲染      原位叠加绘制 ─┘  ← 由 QTimer 驱动，必须在主线程

关于线程模型（与开发文档的差异说明）
    开发文档描述为"多进程 + 队列"。实现时改为**多线程 + 队列**，原因：

    1. ``onnxruntime``（OCR）与 ``llama.cpp``（推理）在计算时都会释放
       GIL，因此线程同样能获得真正的多核并行，CPU 密集型阶段不会互相阻塞；
    2. Windows 上 ``multiprocessing`` 使用 spawn 方式启动子进程，
       每次启动都要重新导入 PyQt6 与全部依赖（数秒），且模型权重需要在
       每个子进程中各加载一份，内存开销成倍增加——这与"8GB 内存可运行"
       的产品目标直接冲突；
    3. 子进程与 Qt 主进程之间传递 numpy 图像需要序列化拷贝，
       在 15fps 下会带来可观的 CPU 与内存带宽开销。

    队列解耦、背压与丢帧策略完全保留，因此架构的分层与解耦特性不变。

背压与时效性
    三级队列全部采用"丢旧保新"。当 OCR 或推理成为瓶颈时，队列会丢弃
    来不及处理的旧帧，保证任何时刻渲染出的译文都尽可能对应当前屏幕内容。
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal

from ..components.capture import CaptureFrame, FrameDiffer, ScreenCapturer
from ..components.capture.geometry import Rect
from ..components.ocr import OcrEngine, RecognizedFrame
from ..components.render import OverlayManager, RenderFrame
from ..components.render.render_item import merge_items, retain_confirmed_items
from ..components.render.space_occupancy import ObservedSpaceDetector
from ..constants import RENDER_POLL_INTERVAL_MS, SKIP_NOTICE_COOLDOWN
from ..utils.execution_evidence import delivery_evidence, trace_translation
from ..utils.logger import get_logger
from .coordinate import CoordinateMapper
from .task_queue import LatestQueue, RateLimiter

logger = get_logger(__name__)


@dataclass
class PipelineStatus:
    """流水线运行状态快照。"""

    running: bool = False
    capture_fps: float = 0.0
    ocr_fps: float = 0.0
    render_fps: float = 0.0
    frames_captured: int = 0
    frames_skipped: int = 0
    frames_recognized: int = 0
    frames_translated: int = 0
    #: 各阶段最近耗时（秒）
    capture_cost: float = 0.0
    ocr_cost: float = 0.0
    translate_cost: float = 0.0
    render_cost: float = 0.0
    #: 端到端延迟（最后一帧从捕获到渲染完成的耗时，秒）
    end_to_end_latency: float = 0.0
    #: 提交绘制与实际绘制分别统计；隐藏覆盖层时只有提交指标。
    submit_latency: float = 0.0
    observation_latency: float = 0.0
    detection_wait_bound: float = 0.0
    latency_scope: str = "initial"
    latency_route: str = "model"
    presentation_confirmed: bool = False
    monitor_interval: float = 0.0
    diagnostics: dict[str, int] = field(default_factory=dict)
    #: 队列深度
    frame_queue_depth: int = 0
    ocr_queue_depth: int = 0
    result_queue_depth: int = 0
    pending_translations: int = 0
    retry_exhausted: bool = False
    untranslated_regions: int = 0

    def as_text(self) -> str:
        text = (
            f"捕获 {self.capture_fps:.0f}fps · OCR {self.ocr_fps:.0f}fps · "
            f"渲染 {self.render_fps:.0f}fps · "
            + (f"显示延迟 ≤{self.end_to_end_latency * 1000:.0f}ms"
               if self.presentation_confirmed else f"提交 {self.submit_latency * 1000:.0f}ms")
        )
        if self.pending_translations:
            text += f" · 待补译 {self.pending_translations} 处"
        if self.untranslated_regions:
            text += f" · 暂未译出 {self.untranslated_regions} 处"
        for key, label in (("ocr_recognized", "识别"), ("ocr_uncertain", "识别疑问"),
                           ("filtered", "过滤"), ("translation_failed", "翻译未完成"),
                           ("review_rejected", "审查未过"), ("display_overflow", "仅文字栏"),
                           ("source_unchanged", "保留原文"), ("review_skipped", "未二审")):
            if self.diagnostics.get(key, 0):
                text += f" · {label} {self.diagnostics[key]} 处"
        return text


@dataclass
class _StageTiming:
    """阶段耗时统计（滑动平均，避免数值剧烈跳动）。"""

    last: float = 0.0
    average: float = 0.0
    samples: int = 0

    def record(self, value: float) -> None:
        self.last = value
        self.samples += 1
        # 指数滑动平均，对近期样本更敏感
        alpha = 0.2 if self.samples > 5 else 1.0 / self.samples
        self.average = value if self.samples == 1 else (self.average * (1 - alpha) + value * alpha)


class Pipeline(QObject):
    """四级异步流水线。

    该对象由 Qt 主线程持有：渲染阶段依赖 QTimer，而 Qt 控件只能在主线程更新。
    捕获、OCR、翻译三个阶段的耗时操作运行在各自的工作线程中。

    :param config: :class:`~pingyitong.config.schema.AppConfig` 配置对象。
        **注意不是** ``SettingsManager``——流水线只需读取配置值，
        而配置变更后 ``SettingsManager`` 会换上一个新的配置对象，
        因此由 :class:`~pingyitong.ui.app.Application` 通过
        :meth:`apply_config` 把新对象推送进来。
    """

    #: 一帧翻译并渲染完成
    frame_rendered = pyqtSignal(object)
    #: 运行状态更新（约每 500ms 一次）
    status_updated = pyqtSignal(object)
    #: 发生错误（供界面提示）
    error_occurred = pyqtSignal(str)
    #: 捕获线程连续失败退出后发出；界面层据此整条停机（状态位与实际一致）
    capture_failed = pyqtSignal()
    #: 译文的完整快照（已按画面差分合并），供实时文本框显示。
    #:
    #: 与 ``frame_rendered`` 的区别：那个是"这一帧识别到的部分"（增量帧只有
    #: 一小块），本信号是"当前屏幕上应有的全部译文"，因此文本框能完整呈现，
    #: 不会随着差分刷新而缺行。
    transcript_updated = pyqtSignal(object)
    #: 整帧都无需翻译，参数为 ``(原因, 目标语言代码)``。
    #:
    #: 用于回答用户最容易困惑的一种情况："我点了开始翻译，屏幕上却
    #: 什么都没变"。最常见的原因是屏幕文字本来就是目标语言——
    #: 此时程序按设计跳过全部段落（原文照原样画回原位），外观上与
    #: 卡死无法区分。这条信号让界面能直接说明原因，而不是让用户猜。
    skip_notice = pyqtSignal(str, str)

    def __init__(
        self,
        config: object,
        capturer: ScreenCapturer,
        differ: FrameDiffer,
        ocr_engine: OcrEngine,
        translator: object,
        overlay: OverlayManager,
        resource_monitor: object | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._capturer = capturer
        self._differ = differ
        self._ocr = ocr_engine
        self._translator = translator
        self._overlay = overlay
        self._monitor = resource_monitor
        # 捕获限速器在 _capture_loop 里创建；挂到实例上让 apply_config
        # 能在运行中热更新帧率（否则用户改"捕获帧率"要重开翻译才生效）。
        self._capture_limiter: RateLimiter | None = None
        # Bounded pixel evidence for read-only expansion: contrasting ink that
        # OCR never returned must not be painted over. Runs in the translate
        # thread, never on the GUI thread.
        self._space_detector = ObservedSpaceDetector()

        # 三级队列
        self._frame_queue: LatestQueue[CaptureFrame] = LatestQueue(
            maxsize=2, keep_latest_only=True, name="frame", coalesce=_coalesce_capture
        )
        self._ocr_queue: LatestQueue[RecognizedFrame] = LatestQueue(
            maxsize=3, keep_latest_only=True, name="ocr", coalesce=_coalesce_recognized
        )
        self._result_queue: LatestQueue[RenderFrame] = LatestQueue(
            maxsize=3, keep_latest_only=True, name="result", coalesce=_coalesce_render
        )

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        #: 注意：``_stop_event`` 是"当前这一代"工作线程的事件，
        #: 每次 :meth:`start` 都会换成新实例。工作线程持有自己出生时的
        #: 那个事件并循环检查它——因此 stop→start 快速交替时，上一代
        #: 卡在长推理中的线程仍会在完成当前条目后退出，不会因为
        #: 新一代重置了事件而复活（复活会变成与新一代并存的僵尸线程，
        #: 双倍推理负载且争抢同一批队列）。
        self._differ_lock = threading.Lock()
        self._previous_frame: CaptureFrame | None = None
        # A language/API switch invalidates work already inside OCR or model
        # inference. Queue clearing alone cannot stop those in-flight calls.
        self._translation_epoch = 0
        self._scene_generation = 0
        self._latest_observation: CaptureFrame | None = None
        self._pending_output_changes: list[Rect] = []
        self._clear_scene_output = False
        self._latency_samples: list[dict] = []
        self._monitor_intervals: list[float] = []
        self._session_started_at = 0.0
        self._observation_epoch_started_at = 0.0
        presented = getattr(type(overlay), "presented", None)
        if presented is not None:
            overlay.presented.connect(self._on_presented)

        #: 框选区域的逻辑坐标 -> 物理坐标换算。显示器几何变化后需刷新。
        self._mapper = CoordinateMapper()

        #: 实时文本框是否正在显示。关闭时不广播文本框更新。
        self._transcript_enabled = False
        #: 本程序自身界面的物理坐标区域（悬浮栏、文本框），OCR 结果落在
        #: 其内的一律丢弃，避免自我反馈。由 UI 层在移动 / 展开时更新。
        self._ignore_regions: list[Rect] = []
        #: 当前屏幕上应有的全部译文（增量帧合并后的结果）
        self._transcript_items: list = []
        #: 最近一次已渲染的完整译文，供复制和重新打开文本框使用。
        self._screen_items: list = []
        self._ocr_context_boxes: list[Rect] = []
        self._pending_retry_regions: list[Rect] = []
        self._unresolved_regions: list[Rect] = []
        self._retry_ready_at = 0.0
        self._retry_attempts = 0
        self._retry_counts: dict[Rect, int] = {}
        self._unresolved_since: dict[Rect, tuple[float, float]] = {}
        self._retry_generation = 0
        self._source_fingerprints: dict[tuple[str, Rect], bytes] = {}

        #: 上一次提示过的"无需翻译"结论，用于去重
        self._last_skip_notice: tuple[str, str] | None = None
        #: 上一次提示的时刻，用于冷却
        self._skip_notice_at = 0.0
        #: 框选区域是否整体落在虚拟桌面之外（显示器断开），只在状态
        #: 翻转时提示一次，避免每帧重复刷日志与通知
        self._region_offscreen = False
        #: 框选区域是否退化为 0×0（悬空配置/映射被裁成空），只在状态
        #: 翻转时记一次 debug 日志
        self._region_degenerate = False

        self._status = PipelineStatus()
        self._timings = {
            "capture": _StageTiming(),
            "ocr": _StageTiming(),
            "translate": _StageTiming(),
            "render": _StageTiming(),
        }
        self._fps_counters = {
            "capture": _FpsCounter(),
            "ocr": _FpsCounter(),
            "render": _FpsCounter(),
        }
        self._last_render_seq = 0

        # 渲染阶段：主线程 QTimer 轮询结果队列
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(RENDER_POLL_INTERVAL_MS)
        self._render_timer.timeout.connect(self._render_tick)

        # 状态上报定时器
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._emit_status)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """流水线是否正在运行。"""
        return self._status.running

    @property
    def status(self) -> PipelineStatus:
        """当前状态快照。"""
        return self._status

    def start(self) -> bool:
        """启动流水线。

        :return: 启动成功返回 ``True``；OCR 组件缺失时返回 ``False``
            并发出错误信号（界面据此提示安装依赖）。
        """
        if self._status.running:
            return True

        preflight = getattr(self._ocr, "preflight_error", None)
        ocr_error = preflight() if callable(preflight) else (
            self._ocr.error_message() or "未安装 RapidOCR" if not self._ocr.is_ready() else ""
        )
        if ocr_error:
            message = f"OCR 组件不可用：{ocr_error}"
            logger.error(message)
            self.error_occurred.emit(message)
            return False

        # 新一代事件：上一代线程持有旧事件，stop 置位后它们照常退出，
        # 不受这里换新事件的影响（见 __init__ 中关于僵尸线程的说明）。
        self._stop_event = threading.Event()
        # 状态重置必须在 _differ_lock 内：stop(0.5s join) 超时后旧代线程
        # 可能仍卡在 _should_process 的锁内操作里，快速 stop->start 切换
        # 时无锁重置会与它们竞争（发布审查确认的窄竞态）。
        with self._differ_lock:
            self._previous_frame = None
            self._latest_observation = None
            self._scene_generation += 1
            self._pending_output_changes = []
            self._clear_scene_output = False
            self._session_started_at = time.perf_counter()
            self._observation_epoch_started_at = self._session_started_at
            self._last_render_seq = 0
            self._ocr_context_boxes = []
            self._pending_retry_regions = []
            self._unresolved_regions = []
            self._retry_attempts = 0
            self._retry_counts = {}
            self._unresolved_since = {}
            self._retry_generation += 1
            self._source_fingerprints = {}
            self._reset_counters()

        stop_event = self._stop_event
        self._spawn("CaptureThread", self._capture_loop, stop_event)
        self._spawn("OcrThread", self._ocr_loop, stop_event)
        self._spawn("TranslateThread", self._translate_loop, stop_event)

        self._render_timer.start()
        self._status_timer.start()
        self._status.running = True
        logger.info(
            "流水线已启动（监测间隔 %.0fms，模型 %s）",
            1000.0 / self._monitor_fps(),
            self._translator.last_stats.model_id or "待定",
        )
        return True

    def stop(self, join_timeout: float = 0.5) -> None:
        """停止流水线并等待各阶段线程退出。

        :param join_timeout: 等待单个线程退出的最长时间（秒）。

        本方法由界面主线程调用，等待时间直接等于界面"未响应"的时间，
        因此默认只做短等待：OCR / 翻译线程可能正卡在一次数秒的推理里，
        等满它们等于把卡顿搬给界面。超时后线程仍在收尾的，靠"线程持有
        自己那一代已置位的停止事件"保证它完成当前条目后必然退出，
        这里只保留引用并记日志供排查。
        """
        if not self._status.running and not self._threads:
            return

        logger.info("正在停止流水线…")
        self._stop_event.set()
        # QTimer 只能在其所属线程启停。stop() 正常由界面主线程调用；
        # 万一从工作线程被调（如 capture_failed 直连槽），跨线程
        # killTimer 会踩 Qt 未定义行为，此处仅在所属线程停表，
        # 异线程调用方只停线程与队列（r0005: PyQt6 qFatal 实测）。
        if QThread.currentThread() is self.thread():
            self._render_timer.stop()
            self._status_timer.stop()

        # 唤醒可能阻塞在队列上的线程
        for queue in (self._frame_queue, self._ocr_queue, self._result_queue):
            queue.wake_all()

        current = threading.current_thread()
        for thread in self._threads:
            if thread is current:
                # 工作线程经信号触发停机时不能 join 自己
                # （RuntimeError 穿过 PyQt6 信号边界 → qFatal → 0xC0000409）
                continue
            thread.join(timeout=join_timeout)

        survivors = [thread for thread in self._threads if thread.is_alive()]
        if survivors:
            logger.warning(
                "流水线线程 %s 仍在收尾（正在处理最后一帧），完成后将自行退出",
                "、".join(thread.name for thread in survivors),
            )
        self._threads = survivors

        self._frame_queue.clear()
        self._ocr_queue.clear()
        self._result_queue.clear()

        self._status.running = False
        self._status.pending_translations = 0
        self._status.retry_exhausted = False
        self._status.untranslated_regions = 0
        self._overlay.clear()
        logger.info("流水线已停止")

    def shutdown(self) -> None:
        """彻底释放资源。"""
        self.stop()
        self._overlay.shutdown()
        self._capturer.close()

    def _spawn(self, name: str, target, stop_event: threading.Event) -> None:
        """创建并启动一个工作线程。

        线程通过参数持有**自己那一代**的停止事件，而不是循环读取
        ``self._stop_event``：后者会被下一次 :meth:`start` 换成新实例，
        若线程读的是共享属性，旧线程会在新一代开始后误以为没有停止
        请求，永远退出不了。
        """
        thread = threading.Thread(
            target=target, args=(stop_event,), name=name, daemon=True
        )
        thread.start()
        self._threads.append(thread)

    def _reset_counters(self) -> None:
        """重置统计计数。"""
        self._status = PipelineStatus(running=False)
        self._latency_samples = []
        self._monitor_intervals = []
        for timing in self._timings.values():
            timing.last = 0.0
            timing.average = 0.0
            timing.samples = 0
        for counter in self._fps_counters.values():
            counter.reset()
        for queue in (self._frame_queue, self._ocr_queue, self._result_queue):
            queue.reset_stats()

    # ------------------------------------------------------------------
    # 阶段一：捕获
    # ------------------------------------------------------------------

    def _capture_loop(self, stop_event: threading.Event) -> None:
        """捕获线程主体。"""
        try:
            self._capturer.open()
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"屏幕捕获初始化失败：{exc}")
            logger.exception("屏幕捕获初始化失败")
            return

        limiter = RateLimiter(self._monitor_fps())
        self._capture_limiter = limiter
        failures = 0

        try:
            while not stop_event.is_set():
                if not limiter.wait(stop_event):
                    break

                try:
                    epoch = self._translation_epoch
                    frame = self._grab_frame()
                except Exception:  # noqa: BLE001
                    failures += 1
                    logger.exception("屏幕捕获失败")
                    if failures >= 5:
                        self.error_occurred.emit("屏幕捕获连续失败，已停止")
                        # 捕获线程就此退出；OCR/翻译线程仍在空转。发专用
                        # 信号让界面在主线程整条停机，_status.running 与
                        # 实际一致（发布审查发现的 minor 状态不一致）。
                        self.capture_failed.emit()
                        break
                    time.sleep(0.2)
                    continue

                failures = 0
                if frame is None:
                    # 退化的框选区域：本帧不抓不译（_grab_frame 内已降频记录）
                    self._status.frames_skipped += 1
                    continue
                frame.__dict__["_translation_epoch"] = epoch
                frame.__dict__["_capture_cost"] = float(
                    getattr(frame, "capture_cost", self._capturer.last_capture_cost)
                )
                self._status.frames_captured += 1
                self._fps_counters["capture"].tick()
                self._timings["capture"].record(self._capturer.last_capture_cost)

                # 画面差分：无变化则跳过后续全部阶段
                if not self._should_process(frame):
                    self._status.frames_skipped += 1
                    continue

                self._frame_queue.put(frame, seq=frame.seq)
        finally:
            self._capturer.close()
            logger.debug("捕获线程已退出")

    def _grab_frame(self) -> CaptureFrame | None:
        """按配置捕获一帧；框选区域退化时返回 ``None``（本帧不抓不译）。"""
        capture_config = self._config.capture

        if capture_config.mode == "region" and capture_config.region:
            # 配置里存的是**逻辑坐标**（框选覆盖层给出的就是逻辑坐标），
            # 而 capture_region 要的是**物理坐标**。这里必须换算：
            # 150% 缩放的主屏上，直接把逻辑值当物理用会少截一大圈
            # （实测 1068×852 逻辑 应为 1602×1278 物理），
            # 于是"框选了一块、实际翻译的是另一块"。
            logical = Rect.from_sequence(capture_config.region)
            physical = self._mapper.capture_region_physical(logical)
            if logical.is_empty or physical.is_empty:
                # 悬空/退化的框选区域（宽高为 0：配置残留或映射结果被
                # 裁剪成空）。mss 对 0×0 区域会直接报错，即便截到空图
                # OCR 与翻译也要白跑一整轮——直接跳过本帧（不抓不译）。
                # 日志只在状态翻转时记一次，避免按帧率刷屏。
                if not self._region_degenerate:
                    self._region_degenerate = True
                    logger.debug(
                        "框选区域 %s 已退化为 0×0，暂停区域捕获",
                        capture_config.region,
                    )
                return None
            if self._region_degenerate:
                self._region_degenerate = False
                logger.debug("框选区域已恢复有效尺寸，恢复区域捕获")
            if self._region_is_offscreen(physical):
                # 框选区域所在的显示器已断开（或区域整体被移出虚拟桌面）。
                # mss 对虚拟桌面之外的区域不会报错，而是"成功"返回一张
                # 全黑位图：差分永远无变化，翻译就此悄悄停摆——状态栏
                # 帧率与延迟一切正常，屏幕上却再无译文，极难定位。
                # 与 capture_monitor 的"显示器序号不存在回退主屏"同理，
                # 这里回退为整屏捕获，并明确告知用户重新框选。
                if not self._region_offscreen:
                    self._region_offscreen = True
                    logger.warning(
                        "框选区域 %s 已在虚拟桌面之外（显示器断开？），回退为整屏捕获",
                        capture_config.region,
                    )
                    self.skip_notice.emit(
                        "region_offscreen",
                        ",".join(str(v) for v in capture_config.region),
                    )
                return self._capturer.capture_all()
            if self._region_offscreen:
                self._region_offscreen = False
                logger.info("框选区域已重新落回虚拟桌面，恢复区域捕获")
            # 同时带上"这块区域属于哪块屏幕"：显示器序号会一路传到渲染阶段，
            # 而渲染阶段按显示器分发条目。缺了它，条目会被当成"不属于任何被
            # 捕获的显示器"而全部丢弃——表现就是翻译在跑、翻译层一片空白。
            return self._capturer.capture_region(
                physical,
                monitor_index=self._mapper.monitor_index_for_physical(physical),
            )

        monitors = capture_config.monitors
        if monitors and len(monitors) == 1:
            return self._capturer.capture_monitor(monitors[0])

        # 全屏或多屏：捕获整个虚拟桌面，差分与识别后按坐标分发到各屏
        return self._capturer.capture_all()

    def _region_is_offscreen(self, physical: Rect) -> bool:
        """框选区域是否整体落在当前虚拟桌面之外。

        显示器断开后，配置里存的区域坐标可能指向一块已不存在的屏幕。
        mss 对这种区域不报错，而是返回全黑位图——必须在捕获前拦下。
        映射器还没有枚举到屏幕（刚启动）时按"在屏内"处理，交由后续
        枚举完成后的刷新去纠正。
        """
        desktop = getattr(self._mapper, "virtual_desktop_physical", None)
        if desktop is None:
            return False
        virtual = desktop() if callable(desktop) else desktop
        if virtual is None or virtual.is_empty:
            return False
        return not physical.intersects(virtual)

    def refresh_screens(self) -> None:
        """重新枚举显示器，刷新坐标换算。

        显示器增删、分辨率或缩放变化后必须调用，否则框选区域仍按
        旧的缩放比例换算，捕获范围会错位。
        """
        self._mapper.refresh()
        self.invalidate()

    def _should_process(self, frame: CaptureFrame) -> bool:
        """判断该帧是否值得进入 OCR 阶段。"""
        if not self._is_current_frame(frame):
            return False
        with self._differ_lock:
            if not self._is_current_frame(frame):
                return False
            previous = self._previous_frame
            self._previous_frame = frame
            self._latest_observation = frame
            frame.__dict__["_scene_generation"] = self._scene_generation
            frame.__dict__["_change_started_at"] = frame.timestamp
            frame.__dict__["_observation_lower_bound"] = (
                previous.timestamp if previous is not None
                else self._observation_epoch_started_at or frame.timestamp
            )
            if previous is not None:
                interval = max(0.0, frame.timestamp - previous.timestamp)
                self._status.monitor_interval = interval
                self._monitor_intervals.append(interval)
                del self._monitor_intervals[:-2000]

        if not self._config.capture.enable_diff:
            return True

        if previous is None:
            # 首帧必须完整识别，否则屏幕上已存在的内容不会被翻译
            frame.is_full_frame = True
            frame.changed_regions = []
            return True

        self._differ.configure(
            pixel_threshold=self._config.capture.diff_pixel_threshold,
            min_area_ratio=self._config.capture.diff_min_area_ratio,
            block_size=self._config.capture.diff_block_size,
        )
        geometry_changed = (
            getattr(previous, "origin", None) != getattr(frame, "origin", None)
            or getattr(previous, "size", None) != getattr(frame, "size", None)
        )
        if geometry_changed:
            from ..components.capture.frame_diff import DiffResult
            result = DiffResult.full_frame(frame.width, frame.height)
        else:
            result = self._differ.compute(previous.image, frame.image)

        # Large changes invalidate the scene, not every capture sequence. Slow
        # translation of stable text may finish despite an unrelated cursor.
        ratio = getattr(result, "change_ratio", 0.0)
        large_change = (
            geometry_changed
            or (isinstance(ratio, (int, float)) and ratio >= 0.35)
        )
        if result.changed:
            with self._differ_lock:
                if large_change:
                    self._scene_generation += 1
                    frame.__dict__["_scene_generation"] = self._scene_generation
                    self._clear_scene_output = True
                    self._ocr_context_boxes = []
                    self._pending_retry_regions = []
                    self._unresolved_regions = []
                    self._retry_counts = {}
                    self._unresolved_since = {}
                self._pending_output_changes.extend(
                    Rect(x, y, w, h).to_global(frame.origin)
                    for x, y, w, h in result.regions
                )
                self._pending_output_changes = list(dict.fromkeys(self._pending_output_changes))

        if not result.changed:
            with self._differ_lock:
                if not self._pending_retry_regions or time.perf_counter() < self._retry_ready_at:
                    return False
                regions = self._pending_retry_regions
                self._pending_retry_regions = []
                for region in regions:
                    self._retry_counts[region] = self._retry_counts.get(region, 0) + 1
                self._retry_attempts = max(self._retry_counts.values(), default=0)
                frame.__dict__["_translation_retry"] = True
                frame.__dict__["_retry_generation"] = self._retry_generation
                self._carry_retry_wait(frame, regions)
            frame.is_full_frame = False
            frame.changed_regions = regions
            return True

        frame.is_full_frame = large_change
        dirty = [Rect(x, y, w, h) for x, y, w, h in result.regions]
        with self._differ_lock:
            source_changed = not self._unresolved_regions or any(
                area.intersects(region) for area in dirty for region in self._unresolved_regions
            )
            if source_changed:
                self._retry_generation += 1
                self._retry_counts = {
                    region: count for region, count in self._retry_counts.items()
                    if not any(region.intersects(area) for area in dirty)
                }
                self._unresolved_since = {
                    region: started for region, started in self._unresolved_since.items()
                    if not any(region.intersects(area) for area in dirty)
                }
                self._retry_attempts = max(self._retry_counts.values(), default=0)
                self._pending_retry_regions = [
                    region for region in self._pending_retry_regions
                    if not any(region.intersects(area) for area in dirty)
                ]
            if self._pending_retry_regions and time.perf_counter() >= self._retry_ready_at:
                self._carry_retry_wait(frame, self._pending_retry_regions)
                dirty.extend(self._pending_retry_regions)
                for region in self._pending_retry_regions:
                    self._retry_counts[region] = self._retry_counts.get(region, 0) + 1
                self._pending_retry_regions = []
                self._retry_attempts = max(self._retry_counts.values(), default=0)
                frame.__dict__["_translation_retry"] = True
                frame.__dict__["_retry_generation"] = self._retry_generation
            known_boxes = [*self._ocr_context_boxes, *(item.box for item in self._screen_items)]
        frame.changed_regions = _expand_text_regions(
            dirty,
            known_boxes, frame.origin, Rect(0, 0, frame.width, frame.height),
        )
        if large_change:
            frame.changed_regions = []
        return True

    def _monitor_fps(self) -> float:
        """Monitoring is independent of the legacy OCR/capture FPS slider."""
        interval = max(10.0, float(getattr(self._config.capture, "monitor_interval_ms", 50)))
        return 1000.0 / interval

    def _carry_retry_wait(self, frame: object, regions: list[Rect]) -> None:
        """A failed region's retry remains part of the same user-visible wait."""
        starts = [self._unresolved_since[area] for area in regions if area in self._unresolved_since]
        if starts:
            frame.__dict__["_change_started_at"] = min(
                float(frame.__dict__["_change_started_at"]), *(value[0] for value in starts)
            )
            frame.__dict__["_observation_lower_bound"] = min(
                float(frame.__dict__["_observation_lower_bound"]), *(value[1] for value in starts)
            )

    # ------------------------------------------------------------------
    # 阶段二：OCR
    # ------------------------------------------------------------------

    def _ocr_loop(self, stop_event: threading.Event) -> None:
        """OCR 线程主体。"""
        logger.debug("OCR 线程已启动")
        reported_ocr_error = ""
        try:
            while not stop_event.is_set():
                frame = self._frame_queue.get(timeout=0.1)
                if frame is None:
                    continue
                if not self._is_current_frame(frame):
                    continue

                # A running app may switch to a missing or unsupported profile.
                # Report it through the existing UI error signal, from this
                # worker; avoid constructing ONNX sessions during configure().
                if not self._ocr.is_ready():
                    error = self._ocr.error_message() or "OCR 模型不可用"
                    if error != reported_ocr_error:
                        self.error_occurred.emit(f"OCR 组件不可用：{error}")
                        reported_ocr_error = error
                    continue
                reported_ocr_error = ""

                started = time.perf_counter()
                try:
                    ocr_started = time.perf_counter()
                    frame_queue_wait = max(
                        0.0,
                        ocr_started
                        - frame.timestamp
                        - float(frame.__dict__.get("_capture_cost", getattr(frame, "capture_cost", 0.0))),
                    )
                    recognized = self._ocr.recognize_frame(frame)
                    recognized.__dict__["_capture_image"] = getattr(frame, "image", None)
                    _carry_observation(frame, recognized)
                except Exception:  # noqa: BLE001 - 单帧识别异常不应终止线程
                    logger.exception("OCR 识别失败，已跳过该帧")
                    continue

                # 识别耗时可能超过 stop 的等待上限；此时本代已被叫停，
                # 识别结果属于过期帧，直接丢弃而不是塞进队列——
                # 否则新一代启动后渲染定时器会把它当新结果画出来。
                if stop_event.is_set():
                    break
                if not self._is_current_frame(frame):
                    continue

                cost = time.perf_counter() - started
                self._timings["ocr"].record(cost)
                self._status.ocr_cost = self._timings["ocr"].average
                self._status.frames_recognized += 1
                self._fps_counters["ocr"].tick()

                # 丢弃落在本程序自身界面（悬浮栏、文本框）内的识别结果。
                # 这些窗口就显示在被捕获的屏幕上，不排除的话它们的文字
                # 会被 OCR → 翻译 → 再次显示，形成自我反馈的乱码循环
                # （实测文本框里出现"宫廷所AD"一类内容即源于此）。
                if self._ignore_regions:
                    recognized.blocks = [
                        block
                        for block in recognized.blocks
                        if not any(
                            block.box.intersects(region)
                            for region in self._ignore_regions
                        )
                    ]

                # 空白结果也必须往下传：画面删除/移动文字后需要清旧译文。
                # 捕获区域相对图像原点，后续各层统一使用全局物理坐标。
                recognized.changed_regions = [
                    region.to_global(recognized.origin)
                    for region in getattr(frame, "changed_regions", [])
                ]
                self._prune_stale_regions(recognized)
                with self._differ_lock:
                    if not self._is_current_frame(frame):
                        continue
                    old_boxes = [] if recognized.is_full_frame else [
                        box for box in self._ocr_context_boxes
                        if not any(box.intersects(area) for area in recognized.changed_regions)
                    ]
                    self._ocr_context_boxes = list(dict.fromkeys([
                        *old_boxes, *(block.box for block in recognized.blocks),
                    ]))
                for key in ("_translation_retry", "_retry_generation"):
                    if key in frame.__dict__:
                        recognized.__dict__[key] = frame.__dict__[key]

                # 记录捕获时刻，用于端到端延迟统计
                recognized.__dict__["_captured_at"] = frame.timestamp
                recognized.__dict__["_capture_image"] = getattr(frame, "image", None)
                recognized.__dict__["_capture_cost"] = float(frame.__dict__.get(
                    "_capture_cost", getattr(frame, "capture_cost", 0.0)
                ))
                recognized.__dict__["_frame_queue_wait"] = frame_queue_wait
                recognized.__dict__["_ocr_finished_at"] = time.perf_counter()
                recognized.__dict__["_translation_epoch"] = frame.__dict__.get(
                    "_translation_epoch", self._translation_epoch
                )
                self._ocr_queue.put(recognized, seq=recognized.seq)
        finally:
            logger.debug("OCR 线程已退出")

    # ------------------------------------------------------------------
    # 阶段三：翻译
    # ------------------------------------------------------------------

    def _attach_space_evidence(self, recognized: object, render_frame: object) -> float:
        """Attach bounded pixel barriers so expansion never covers unread ink.

        Sources are every OCR block of this frame plus the paragraph sources the
        renderer knows about; their own pixels are excluded from the barrier
        search, so a detected source is never forbidden to keep its anchor.
        Returns the measured cost in seconds for latency accounting.
        """
        image = getattr(render_frame, "__dict__", {}).get("_capture_image")
        origin = getattr(recognized, "origin", None)
        if image is None or origin is None:
            render_frame.__dict__["_space_obstacles"] = None
            return 0.0
        sources = [block.box for block in getattr(recognized, "blocks", []) or []]
        sources.extend(
            item.box for item in (getattr(render_frame, "source_items", None) or [])
        )
        started = time.perf_counter()
        try:
            observation = self._space_detector.analyze(image, origin, sources)
        except Exception:  # noqa: BLE001 - layout must not fail on evidence
            logger.exception("空间占用观测失败，本帧退回仅按源框排版")
            render_frame.__dict__["_space_obstacles"] = None
            return time.perf_counter() - started
        cost = time.perf_counter() - started
        render_frame.__dict__["_space_obstacles"] = list(observation.obstacles)
        render_frame.__dict__["_space_evidence"] = {
            **observation.evidence,
            "source_count": len(sources),
            "constraint": "expansion_only",
        }
        render_frame.timings["space_observe"] = cost
        return cost

    def _translate_loop(self, stop_event: threading.Event) -> None:
        """翻译线程主体。"""
        logger.debug("翻译线程已启动")
        try:
            while not stop_event.is_set():
                recognized = self._ocr_queue.get(timeout=0.1)
                if recognized is None:
                    continue
                if not self._is_current_frame(recognized):
                    continue
                self._prune_stale_regions(recognized)

                snapshot = self._monitor.sample() if self._monitor is not None else None
                cpu_sustained = False
                if self._monitor is not None and hasattr(
                    self._monitor, "cpu_above_threshold_for"
                ):
                    cpu_sustained = self._monitor.cpu_above_threshold_for(
                        self._config.schedule.cpu_high_threshold,
                        self._config.schedule.cpu_high_sustain_seconds,
                    )

                started = time.perf_counter()
                ocr_finished_at = recognized.__dict__.get("_ocr_finished_at", started)
                ocr_queue_wait = max(0.0, started - float(ocr_finished_at))
                # 纠错/重试是逐轮的请求；翻译期间新帧到来（invalidate 推进
                # 代数）后继续补救旧帧只会拖慢新帧。回调随帧对象传入，
                # 供翻译服务在每轮纠错前判断是否立即中止。
                recognized.__dict__["_frame_stale"] = (
                    lambda frame=recognized: not self._is_current_frame(frame)
                )
                try:
                    with trace_translation(recognized) as execution_trace:
                        render_frame = self._translator.translate_frame(
                            recognized,
                            snapshot=snapshot,
                            cpu_sustained_high=cpu_sustained,
                        )
                    if execution_trace is not None:
                        render_frame.__dict__["_execution_evidence"] = execution_trace.snapshot()
                except Exception:  # noqa: BLE001 - 单帧翻译异常不应终止线程
                    logger.exception("翻译失败，已跳过该帧")
                    continue

                # 长句推理可能远超 stop 的等待上限（实测可达数十秒）；
                # 本代已被叫停后产出的是过期帧，丢弃而不是入队。
                if stop_event.is_set():
                    break
                if not self._is_current_frame(recognized):
                    continue

                cost = time.perf_counter() - started
                self._timings["translate"].record(cost)
                self._status.translate_cost = self._timings["translate"].average
                self._status.frames_translated += 1

                captured_at = recognized.__dict__.get("_captured_at")
                if captured_at:
                    render_frame.__dict__["_captured_at"] = captured_at
                render_frame.timings.update(
                    {
                        "capture": float(
                            recognized.__dict__.get("_capture_cost", 0.0)
                        ),
                        "frame_queue": float(
                            recognized.__dict__.get("_frame_queue_wait", 0.0)
                        ),
                        "ocr": float(recognized.elapsed),
                        "ocr_queue": ocr_queue_wait,
                    }
                )
                self._attach_space_evidence(recognized, render_frame)
                render_frame.__dict__["_result_enqueued_at"] = time.perf_counter()
                render_frame.__dict__["_translation_epoch"] = recognized.__dict__.get(
                    "_translation_epoch", self._translation_epoch
                )
                render_frame.changed_regions = list(recognized.changed_regions)
                render_frame.__dict__["_capture_monitor_index"] = recognized.monitor_index
                render_frame.__dict__["_capture_image"] = recognized.__dict__.get("_capture_image")
                _carry_observation(recognized, render_frame)
                render_frame.diagnostics = {
                    **getattr(recognized, "diagnostics", {}),
                    **getattr(render_frame, "diagnostics", {}),
                }
                render_frame.__dict__["_diagnostic_events"] = [
                    {"source_seq": recognized.seq, **event}
                    for event in [*getattr(recognized, "decision_events", []),
                                  *getattr(render_frame, "_diagnostic_events", [])]
                ]
                render_frame.__dict__["_source_lines"] = [
                    {"index": block.index, "line_id": block.line_id,
                     "text": block.text, "rect": [block.box.x, block.box.y, block.box.width, block.box.height],
                     "confidence": block.confidence, "ocr_uncertain": block.ocr_uncertain}
                    for block in recognized.blocks
                ]
                for key in ("_translation_retry", "_retry_generation"):
                    if key in recognized.__dict__:
                        render_frame.__dict__[key] = recognized.__dict__[key]
                self._result_queue.put(render_frame, seq=render_frame.seq)
                self._check_skip_notice(render_frame)
        finally:
            logger.debug("翻译线程已退出")

    def _check_skip_notice(self, render_frame: object) -> None:
        """整帧内容没有任何可见变化时，给出一次说明。

        判据是"**译文与原文一样**"而不是"统计上没翻译"：实测遇到过
        17 段里"译了 4 段"、但那 4 段译出来与原文完全相同的情况
        （路径、代码片段这类内容，模型按要求保持原样），
        统计上不算"全部跳过"，用户看到的却依然是"什么都没变"。

        通知带**冷却时间**。只按"结论是否变化"去重是不够的：屏幕上
        只要有一小块内容在变（光标闪烁、时钟走动），结论就会在
        "有变化 / 无变化"之间来回翻转，提示因此反复弹出——实测 9 秒内
        弹了两次。加了冷却之后，同一结论在冷却期内只提示一次；
        而更换目标语言是用户主动操作，属于新结论，立即提示。
        """
        items = [
            item
            for item in getattr(render_frame, "items", [])
            if getattr(item, "is_renderable", False)
        ]
        if not items:
            return

        # 有任意一条内容发生了变化，就说明翻译确实生效了
        if any(_visible_change(item) for item in items):
            return

        target = self._config.translation.target_language
        conclusion = ("already_target", target)
        now = time.perf_counter()

        if (
            conclusion == self._last_skip_notice
            and now - self._skip_notice_at < SKIP_NOTICE_COOLDOWN
        ):
            return

        self._last_skip_notice = conclusion
        self._skip_notice_at = now

        logger.info(
            "本帧 %s 条内容均无可见变化：屏幕文字已是目标语言「%s」或无需翻译",
            len(items),
            target,
        )
        self.skip_notice.emit("already_target", target)

    # ------------------------------------------------------------------
    # 阶段四：渲染（主线程）
    # ------------------------------------------------------------------

    def _render_tick(self) -> None:
        """渲染阶段：在主线程中分发最新结果。

        只取最新一帧：中间帧没有绘制意义，绘制它们只会拖慢界面。

        **两个消费方，各自判断可见性**：原位翻译层隐藏（例如文本框模式
        开启时）**不能**让整个渲染阶段短路——文本框依赖
        :meth:`_update_transcript` 拿到译文，而它就在本方法里。
        曾经在这里直接 ``return``，结果是"开启文本框翻译模式后，
        面板永远停在等待识别"——翻译线程其实一直在产出，
        全部在结果队列里被当作过期帧丢弃了。
        """
        self._clear_changed_output()
        render_frame = self._result_queue.get_latest()
        if render_frame is None or not self._is_current_frame(render_frame):
            return
        self._prune_stale_regions(render_frame)
        render_frame.__dict__["_latency_scope"] = (
            "initial" if self._last_render_seq == 0 or render_frame.is_full_frame else "partial"
        )
        render_frame.__dict__["_refresh_kind"] = (
            "initial" if self._last_render_seq == 0 else
            "full_refresh" if render_frame.is_full_frame else "partial_update"
        )
        accepted = [item for item in render_frame.items
                    if getattr(item, "quality_status", "accepted") == "accepted"]
        cache_hits = sum(bool(getattr(item, "from_cache", False)) for item in accepted)
        render_frame.__dict__["_latency_route"] = (
            "cache" if accepted and cache_hits == len(accepted)
            else "mixed" if cache_hits else "model" if accepted else "uncertain"
        )

        started = time.perf_counter()
        try:
            image = render_frame.__dict__.get("_capture_image")
            sources = getattr(render_frame, "source_items", None)
            if image is not None:
                sources = list(sources or [])
                for old in self._screen_items:
                    if getattr(old, "quality_status", "accepted") != "accepted":
                        continue
                    if any(source.box.intersects(old.box) for source in sources):
                        # A new OCR reading may correct a previously missed
                        # negation even on identical pixels. Its explicit
                        # source evidence outranks the old pixel fingerprint.
                        continue
                    signature = (old.source_text, old.box)
                    previous_fingerprint = self._source_fingerprints.get(signature)
                    if previous_fingerprint is not None and previous_fingerprint == _source_fingerprint(image, old.box, render_frame.origin):
                        # OCR may miss the same small font for one pass. Pixel
                        # evidence allows keeping it without retaining deleted
                        # or changed source text on the basis of a timeout.
                        sources.append(type(old)(box=old.box, text=old.source_text, source_text=old.source_text))
            incoming_count = len(render_frame.items)
            render_frame.items = retain_confirmed_items(
                self._screen_items, render_frame.items,
                sources,
            )
            retained_count = max(0, len(render_frame.items) - incoming_count)
            render_frame.deduplicate()
            render_frame.__dict__["_delivery_evidence"] = delivery_evidence(
                render_frame, retained_count=retained_count,
            )
            active_monitors = None
            monitors_config = self._config.capture.monitors
            if monitors_config:
                active_monitors = set(monitors_config)

            # 整帧识别才替换整层内容；增量帧只更新识别到的区域，
            # 否则未变化区域的译文会逐帧闪空再画回（整屏闪烁）。
            # 翻译层隐藏时跳过分发，但结果仍然进入文本框路径。
            if self._overlay.is_visible:
                dirty_kwargs = (
                    {"dirty_regions": render_frame.changed_regions}
                    if render_frame.changed_regions else {}
                )
                apply_frame = getattr(type(self._overlay), "apply_frame", None)
                apply_method = self._overlay.apply_frame if callable(apply_frame) else self._overlay.apply
                apply_method(
                    render_frame if callable(apply_frame) else render_frame.items,
                    active_monitors,
                    replace=bool(getattr(render_frame, "is_full_frame", True)),
                    **dirty_kwargs,
                )
                self._last_render_seq = render_frame.seq
        except Exception:  # noqa: BLE001 - 渲染异常不应终止定时器
            logger.exception("渲染失败")
            return

        cost = time.perf_counter() - started
        self._timings["render"].record(cost)
        self._status.render_cost = self._timings["render"].average
        self._fps_counters["render"].tick()

        captured_at = render_frame.__dict__.get("_captured_at")
        if captured_at:
            finished_at = time.perf_counter()
            e2e = max(0.0, finished_at - float(captured_at))
            result_enqueued_at = render_frame.__dict__.get(
                "_result_enqueued_at", started
            )
            render_frame.timings["result_queue"] = max(
                0.0, started - float(result_enqueued_at)
            )
            render_frame.timings["render"] = cost
            earliest = float(render_frame.__dict__.get("_change_started_at", captured_at))
            render_frame.timings["capture_to_submit"] = e2e
            render_frame.timings["observed_to_submit"] = max(0.0, finished_at - earliest)
            self._status.submit_latency = render_frame.timings["observed_to_submit"]
            self._status.presentation_confirmed = False
            if render_frame.items:
                known = sum(
                    float(render_frame.timings.get(name, 0.0))
                    for name in (
                        "capture", "frame_queue", "ocr", "ocr_queue",
                        "preprocess", "inference", "space_observe",
                        "result_queue", "render",
                    )
                )
                other = max(0.0, e2e - known)
                logger.info(
                    "提交延迟 %.0fms（%d 条，%s） | "
                    "捕获 %.0f + 帧队列 %.0f + OCR %.0f + OCR队列 %.0f + "
                    "预处理 %.0f + 模型 %.0f + 空间观测 %.0f + 其他 %.0f + "
                    "结果队列 %.0f + 渲染 %.0fms",
                    e2e * 1000, len(render_frame.items),
                    render_frame.__dict__["_latency_route"],
                    render_frame.timings.get("capture", 0.0) * 1000,
                    render_frame.timings.get("frame_queue", 0.0) * 1000,
                    render_frame.timings.get("ocr", 0.0) * 1000,
                    render_frame.timings.get("ocr_queue", 0.0) * 1000,
                    render_frame.timings.get("preprocess", 0.0) * 1000,
                    render_frame.timings.get("inference", 0.0) * 1000,
                    render_frame.timings.get("space_observe", 0.0) * 1000,
                    other * 1000,
                    render_frame.timings.get("result_queue", 0.0) * 1000,
                    render_frame.timings.get("render", 0.0) * 1000,
                )

        self._update_transcript(render_frame)
        self._status.diagnostics = dict(getattr(render_frame, "diagnostics", {}))
        fingerprints = dict(self._source_fingerprints)
        if image is not None:
            for item in render_frame.items:
                fingerprint = _source_fingerprint(image, item.box, render_frame.origin)
                if fingerprint is not None:
                    fingerprints[(item.source_text, item.box)] = fingerprint
        self._source_fingerprints = {
            (item.source_text, item.box): fingerprints[(item.source_text, item.box)]
            for item in self._screen_items if (item.source_text, item.box) in fingerprints
        }
        retry_regions = list(getattr(render_frame, "retry_regions", []))
        with self._differ_lock:
            bounds = Rect(0, 0, render_frame.origin.width, render_frame.origin.height)
            new_unresolved = [
                region.to_local(render_frame.origin).clamp(bounds) for region in retry_regions
                if not region.to_local(render_frame.origin).clamp(bounds).is_empty
            ]
            retained_unresolved = [] if render_frame.is_full_frame else [
                region for region in self._unresolved_regions
                if not any(region.to_global(render_frame.origin).intersects(area) for area in render_frame.changed_regions)
            ]
            self._unresolved_regions = list(dict.fromkeys([*retained_unresolved, *new_unresolved]))
            counts = self._retry_counts
            self._retry_counts = {
                region: max((count for old, count in counts.items()
                             if region.intersection(old).area >= max(region.area, old.area) * 0.75), default=0)
                for region in self._unresolved_regions
            }
            current_start = float(render_frame.__dict__.get("_change_started_at", render_frame.__dict__.get("_captured_at", time.perf_counter())))
            current_lower = float(render_frame.__dict__.get("_observation_lower_bound", current_start))
            previous_starts = self._unresolved_since
            self._unresolved_since = {
                region: min((value for old, value in previous_starts.items()
                             if region.intersection(old).area >= max(region.area, old.area) * 0.75),
                            default=(current_start, current_lower))
                for region in self._unresolved_regions
            }
            eligible = [region for region in self._unresolved_regions if self._retry_counts[region] < 3]
            self._status.pending_translations = len(eligible)
            self._status.untranslated_regions = len(self._unresolved_regions) - len(eligible)
            self._status.retry_exhausted = bool(self._unresolved_regions) and not eligible
            if eligible:
                # Slow inference must not enqueue repeated copies while a
                # retry is in flight. Clear this list when it is dispatched.
                was_pending = bool(self._pending_retry_regions)
                self._pending_retry_regions = eligible
                if new_unresolved or not was_pending:
                    self._retry_ready_at = time.perf_counter() + 2 ** max(self._retry_counts[area] for area in eligible)
            else:
                self._pending_retry_regions = []
        self.frame_rendered.emit(render_frame)
        self._last_render_seq = render_frame.seq

    def _clear_changed_output(self) -> None:
        """Remove obsolete coverage on the GUI thread, before slow OCR returns."""
        with self._differ_lock:
            dirty, self._pending_output_changes = self._pending_output_changes, []
            clear_all, self._clear_scene_output = self._clear_scene_output, False
            latest = self._latest_observation
        if not clear_all and not dirty:
            return
        kept = []
        for item in self._screen_items:
            if clear_all:
                continue
            if not any(item.box.intersects(area) for area in dirty):
                kept.append(item)
                continue
            signature = (item.source_text, item.box)
            old = self._source_fingerprints.get(signature)
            if latest is not None and old is not None and old == _source_fingerprint(
                latest.image, item.box, latest.origin
            ):
                kept.append(item)
        if len(kept) == len(self._screen_items) and not clear_all:
            return
        self._screen_items = kept
        self._source_fingerprints = {
            (item.source_text, item.box): self._source_fingerprints[(item.source_text, item.box)]
            for item in kept if (item.source_text, item.box) in self._source_fingerprints
        }
        if self._overlay.is_visible:
            self._overlay.apply(kept, replace=True)
        if self._transcript_enabled:
            self._transcript_items = list(kept)
            self.transcript_updated.emit(list(kept))

    def _source_still_current(self, frame: object, box: Rect, latest=None) -> bool:
        latest = latest or self._latest_observation
        image = getattr(frame, "__dict__", {}).get("_capture_image")
        if latest is None or image is None or latest.seq <= getattr(frame, "seq", latest.seq):
            return True
        origin = getattr(frame, "origin", None)
        if origin != latest.origin:
            return False
        old = _source_fingerprint(image, box, origin)
        return old is not None and old == _source_fingerprint(latest.image, box, latest.origin)

    def _prune_stale_regions(self, frame: object) -> None:
        """Accept unchanged source regions even when another region has changed."""
        latest = self._latest_observation
        image = getattr(frame, "__dict__", {}).get("_capture_image")
        if latest is None or image is None or latest.seq <= frame.seq:
            return
        def valid(box):
            return self._source_still_current(frame, box, latest)
        full = bool(getattr(frame, "is_full_frame", True))
        frame_box = frame.origin
        if full and valid(frame_box):
            return
        if hasattr(frame, "blocks"):
            frame.blocks = [block for block in frame.blocks if valid(block.box)]
            sources = frame.blocks
        else:
            frame.items = [item for item in frame.items if valid(item.box)]
            if getattr(frame, "source_items", None) is not None:
                frame.source_items = [item for item in frame.source_items if valid(item.box)]
            frame.retry_regions = [area for area in frame.retry_regions if valid(area)]
            sources = [*(getattr(frame, "source_items", None) or []), *frame.items]
        if full:
            # An older full snapshot cannot erase newer regions. Include stable
            # old display boxes so a genuine deletion can still be applied.
            dirty = [item.box for item in sources]
            dirty.extend(item.box for item in self._screen_items if valid(item.box))
        else:
            dirty = [area for area in frame.changed_regions if valid(area)]
            # A dirty paragraph may include a separate changing cursor. Valid
            # individual OCR lines are still useful and should not starve.
            dirty.extend(item.box for item in sources)
        frame.is_full_frame = False
        frame.changed_regions = list(dict.fromkeys(dirty))

    def _on_presented(self, frame: object, painted_at: float) -> None:
        """Acknowledge actual Qt painting, separately from queued UI submission."""
        if not self._is_current_frame(frame):
            return
        captured_at = frame.__dict__.get("_captured_at")
        if captured_at is None:
            return
        observed = float(frame.__dict__.get("_change_started_at", captured_at))
        lower = min(observed, float(frame.__dict__.get("_observation_lower_bound", observed)))
        observation_latency = max(0.0, painted_at - observed)
        wait_bound = max(0.0, observed - lower)
        total_bound = observation_latency + wait_bound
        frame.timings.update({
            "observed_to_present": observation_latency,
            "detection_wait_bound": wait_bound,
            "end_to_end_upper_bound": total_bound,
            "end_to_end": total_bound,
            "presented_at": painted_at,
        })
        self._status.observation_latency = observation_latency
        self._status.detection_wait_bound = wait_bound
        self._status.end_to_end_latency = total_bound
        self._status.presentation_confirmed = True
        self._status.latency_scope = frame.__dict__.get("_latency_scope", "initial")
        self._status.latency_route = frame.__dict__.get("_latency_route", "model")
        self._status.diagnostics = dict(getattr(frame, "diagnostics", {}))
        sample = {
            "seq": frame.seq, "scope": self._status.latency_scope,
            "route": self._status.latency_route, "observed_at": observed,
            "appearance_lower_bound": lower, "presented_at": painted_at,
            "observed_to_present": observation_latency,
            "end_to_end_upper_bound": total_bound,
            "timings": dict(frame.timings),
            "refresh_kind": getattr(frame, "_refresh_kind", None),
            "delivery_evidence": getattr(frame, "_delivery_evidence", None),
        }
        self._latency_samples.append(sample)
        del self._latency_samples[:-2000]
        logger.info(
            "实际绘制 %s/%s：观测后 %.0fms，场景等待上界 %.0fms（发现窗口 %.0fms）",
            sample["scope"], sample["route"], observation_latency * 1000,
            total_bound * 1000, wait_bound * 1000,
        )

    def latency_snapshot(self) -> dict:
        """Bounded raw samples for honest per-scope/per-route acceptance reports."""
        return {
            "clock": "time.perf_counter",
            "appearance_time": "between_previous_observation_and_first_detection",
            "samples": [dict(sample) for sample in self._latency_samples],
            "monitor_intervals": list(self._monitor_intervals),
        }

    def set_ignore_regions(self, regions: list[Rect]) -> None:
        """设置 OCR 结果需要忽略的屏幕区域（物理坐标）。

        传入本程序自身界面（悬浮栏、实时文本框）的当前几何，
        防止它们的文字被识别、翻译后再显示，形成自我反馈。
        """
        self._ignore_regions = list(regions or [])

    def set_transcript_enabled(self, enabled: bool) -> None:
        """开启 / 关闭文本框广播，保留复制需要的完整画面快照。"""
        self._transcript_enabled = bool(enabled)
        if enabled:
            self._transcript_items = list(self._screen_items)
            self.transcript_updated.emit(list(self._transcript_items))
        else:
            self._transcript_items = []

    def snapshot_items(self) -> list:
        """返回已被界面接受的完整译文；在途和失效帧不会进入此快照。"""
        return list(self._screen_items)

    def _update_transcript(self, render_frame: object) -> None:
        """维护"当前屏幕全部译文"的合并快照并广播。

        合并复用渲染层的同一份规则（:func:`merge_items`），避免两处判定
        不一致导致文本框内容与屏幕上的译文对不上。
        """
        items = [
            item
            for item in getattr(render_frame, "items", [])
            if getattr(item, "is_renderable", False)
        ]
        replace = bool(getattr(render_frame, "is_full_frame", True))
        if replace:
            self._screen_items = list(items)
        else:
            self._screen_items = merge_items(
                self._screen_items,
                items,
                getattr(render_frame, "changed_regions", []),
            )
        if self._transcript_enabled:
            self._transcript_items = list(self._screen_items)
            self.transcript_updated.emit(list(self._transcript_items))

    # ------------------------------------------------------------------
    # 状态上报
    # ------------------------------------------------------------------

    def _emit_status(self) -> None:
        """周期性更新并广播状态，同时回填帧率给资源监控。"""
        self._status.capture_fps = self._fps_counters["capture"].fps()
        self._status.ocr_fps = self._fps_counters["ocr"].fps()
        self._status.render_fps = self._fps_counters["render"].fps()

        self._status.frame_queue_depth = len(self._frame_queue)
        self._status.ocr_queue_depth = len(self._ocr_queue)
        self._status.result_queue_depth = len(self._result_queue)

        if self._monitor is not None:
            if hasattr(self._monitor, "report_render_fps"):
                self._monitor.report_render_fps(self._status.render_fps)
            if hasattr(self._monitor, "report_capture_fps"):
                self._monitor.report_capture_fps(self._status.capture_fps)
            # 回填端到端延迟：调度器用它判断"用户是否在等"，
            # 而不是用渲染帧率（静态屏幕下帧率天然趋近 0，会误判为卡顿）
            if hasattr(self._monitor, "report_latency"):
                self._monitor.report_latency(self._status.end_to_end_latency)

        self.status_updated.emit(self._status)

    # ------------------------------------------------------------------
    # 运行时调整
    # ------------------------------------------------------------------

    def apply_config(self, config: object) -> None:
        """应用配置变更。"""
        self._config = config
        if self._capture_limiter is not None:
            self._capture_limiter.configure(self._monitor_fps())
        self._ocr.configure(
            upscale_factor=config.ocr.upscale_factor,
            enable_preprocess=config.ocr.enable_preprocess,
            min_confidence=config.ocr.min_confidence,
            min_box_height=config.ocr.min_box_height,
            box_padding=config.ocr.box_padding,
            threads=config.ocr.threads,
            source_language=config.translation.source_language,
        )

    def invalidate(self, *, clear_output: bool = False) -> None:
        """使下一帧强制整帧识别。

        用于以下场景：翻译层被重新显示、目标语言变更、显示器几何变化——
        这些情况下屏幕内容未变，但差分会让流水线误以为"没有变化"。
        ``clear_output`` 用于翻译方向或推理后端切换：清掉旧译文，
        且让已进入 OCR/模型推理的旧帧在完成后被丢弃。
        """
        with self._differ_lock:
            self._translation_epoch += 1
            self._scene_generation += 1
            # Losing the diff baseline starts a new observation window. The
            # session may have begun minutes ago, before this refresh request.
            self._observation_epoch_started_at = time.perf_counter()
            self._previous_frame = None
            self._latest_observation = None
            self._pending_output_changes = []
            self._clear_scene_output = False
            self._ocr_context_boxes = []
            self._pending_retry_regions = []
            self._unresolved_regions = []
            self._retry_attempts = 0
            self._retry_counts = {}
            self._unresolved_since = {}
            self._retry_generation += 1
            self._source_fingerprints = {}
            self._status.pending_translations = 0
            self._status.retry_exhausted = False
            self._status.untranslated_regions = 0
            if clear_output:
                # Keep a fresh frame from becoming the diff baseline before
                # its queued copy is cleared. A static screen would then be
                # skipped forever without producing the new translation.
                for queue in (self._frame_queue, self._ocr_queue, self._result_queue):
                    queue.clear()
        if clear_output:
            self._screen_items = []
            self._transcript_items = []
            self._overlay.clear()
            if self._transcript_enabled:
                self.transcript_updated.emit([])
        logger.debug("已请求下一帧整帧识别")

    def _is_current_frame(self, frame: object) -> bool:
        """Discard frames captured before the latest invalidation."""
        metadata = getattr(frame, "__dict__", {})
        if metadata.get("_scene_generation", self._scene_generation) != self._scene_generation:
            return False
        current = metadata.get("_retry_generation", self._retry_generation) == self._retry_generation and metadata.get(
            "_translation_epoch", self._translation_epoch
        ) == self._translation_epoch
        if not current:
            return False
        sources = getattr(frame, "blocks", None)
        if sources is None:
            sources = getattr(frame, "source_items", None) or getattr(frame, "items", None)
        return not sources or any(self._source_still_current(frame, source.box) for source in sources)

    def queue_stats(self) -> dict[str, str]:
        """返回各级队列统计，用于性能压测面板。"""
        return {
            "捕获队列": self._frame_queue.stats().as_text(),
            "OCR 队列": self._ocr_queue.stats().as_text(),
            "渲染队列": self._result_queue.stats().as_text(),
        }


def _compatible_snapshots(previous: object, latest: object) -> bool:
    """Merge only pending changes from the same capture space and config epoch."""
    old_meta = getattr(previous, "__dict__", {})
    new_meta = getattr(latest, "__dict__", {})
    return (
        old_meta.get("_translation_epoch") == new_meta.get("_translation_epoch")
        and old_meta.get("_scene_generation") == new_meta.get("_scene_generation")
        and getattr(previous, "origin", None) == getattr(latest, "origin", None)
        and getattr(previous, "size", None) == getattr(latest, "size", None)
        and getattr(previous, "monitor_index", old_meta.get("_capture_monitor_index"))
        == getattr(latest, "monitor_index", new_meta.get("_capture_monitor_index"))
    )


def _carry_observation(source: object, target: object) -> None:
    """Carry immutable capture ownership through asynchronous stages."""
    metadata = getattr(source, "__dict__", {})
    for key in ("_scene_generation", "_change_started_at", "_observation_lower_bound"):
        if key in metadata:
            target.__dict__[key] = metadata[key]


def _merge_observation(previous: object, latest: object) -> None:
    """Coalescing work must not reset the user's outstanding wait."""
    for key in ("_change_started_at", "_observation_lower_bound"):
        old = previous.__dict__.get(key, previous.__dict__.get("_captured_at", getattr(previous, "timestamp", None)))
        new = latest.__dict__.get(key, latest.__dict__.get("_captured_at", getattr(latest, "timestamp", None)))
        if old is not None:
            latest.__dict__[key] = min(float(old), float(new)) if new is not None else float(old)


def _source_fingerprint(image: object, box: Rect, origin: Rect) -> bytes | None:
    """Hash source pixels only; no translated-window pixels or text guessing."""
    if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
        return None
    local = box.to_local(origin).clamp(Rect(0, 0, image.shape[1], image.shape[0]))
    if local.is_empty:
        return None
    patch = image[local.top:local.bottom, local.left:local.right]
    return hashlib.blake2b(patch.tobytes(), digest_size=16).digest()


def _expand_text_regions(
    regions: list[Rect], known_boxes: list[Rect], origin: Rect, bounds: Rect,
) -> list[Rect]:
    """差分命中一小片时重新识别完整原文，防止残片替换整段译文。"""
    boxes = [box.to_local(origin).clamp(bounds) for box in known_boxes]
    expanded = []
    for region in regions:
        region = region.clamp(bounds)
        if region.is_empty:
            continue
        previous = None
        while previous != region:
            previous = region
            for box in boxes:
                if not box.is_empty and region.intersects(box):
                    region = region.union(box).clamp(bounds)
        if region not in expanded:
            expanded.append(region)
    # Keep distant changes separate, but do not recognize overlapping patches
    # twice after expanding a paragraph or a chain of OCR boxes.
    merged: list[Rect] = []
    for region in expanded:
        while any(region.intersects(other) for other in merged):
            overlap = next(other for other in merged if region.intersects(other))
            merged.remove(overlap)
            region = region.union(overlap)
        merged.append(region)
    return merged


def _combine_regions(previous: list[Rect], latest: list[Rect]) -> list[Rect]:
    """Keep separate changed areas: their bounding union could erase unchanged text."""
    return list(dict.fromkeys([*previous, *latest]))


def _coalesce_capture(previous: CaptureFrame, latest: CaptureFrame) -> CaptureFrame:
    """Recognize all pending changes using the newest capture's pixels and time."""
    if not _compatible_snapshots(previous, latest) or getattr(latest, "is_full_frame", True):
        return latest
    _merge_observation(previous, latest)
    if getattr(previous, "is_full_frame", True):
        latest.is_full_frame = True
        latest.changed_regions = []
    else:
        latest.changed_regions = _combine_regions(
            previous.changed_regions, latest.changed_regions
        )
        # Many accumulated tiles are cheaper to recognize in one pass.
        if len(latest.changed_regions) > 64:
            latest.is_full_frame = True
            latest.changed_regions = []
    return latest


def _coalesce_recognized(
    previous: RecognizedFrame, latest: RecognizedFrame,
) -> RecognizedFrame:
    """Carry unprocessed text changes forward while replacing changed/empty regions."""
    if not _compatible_snapshots(previous, latest) or latest.is_full_frame:
        return latest
    _merge_observation(previous, latest)
    latest.decision_events = [*previous.decision_events, *latest.decision_events]
    latest.blocks = [
        old for old in previous.blocks
        if not any(old.box.intersects(area) for area in latest.changed_regions)
        and not any(old.box.intersects(new.box) for new in latest.blocks)
    ] + latest.blocks
    latest.blocks.sort(key=lambda block: (block.box.top, block.box.left))
    for index, block in enumerate(latest.blocks):
        block.index = index
    latest.is_full_frame = previous.is_full_frame
    latest.changed_regions = (
        [] if latest.is_full_frame else _combine_regions(
            previous.changed_regions, latest.changed_regions
        )
    )
    return latest


def _coalesce_render(previous: RenderFrame, latest: RenderFrame) -> RenderFrame:
    """Coalesce untranslated intermediate display updates without losing deletions."""
    if not _compatible_snapshots(previous, latest) or latest.is_full_frame:
        return latest
    _merge_observation(previous, latest)
    latest.__dict__["_diagnostic_events"] = [
        *getattr(previous, "_diagnostic_events", []), *getattr(latest, "_diagnostic_events", []),
    ]
    for key in ("_paragraph_provenance", "_source_lines"):
        incoming = getattr(latest, key, [])
        incoming_rects = [Rect(*entry["rect"]) for entry in incoming if len(entry.get("rect", [])) == 4]
        latest.__dict__[key] = [
            entry for entry in getattr(previous, key, [])
            if len(entry.get("rect", [])) == 4
            and not any(Rect(*entry["rect"]).intersects(rect)
                        for rect in [*latest.changed_regions, *incoming_rects])
        ] + incoming
    previous_trace = getattr(previous, "_execution_evidence", None)
    latest_trace = getattr(latest, "_execution_evidence", None)
    if previous_trace or latest_trace:
        complete = bool(previous_trace and latest_trace
                        and previous_trace.get("completed") and latest_trace.get("completed")
                        and not previous_trace.get("coalesced_unknown") and not latest_trace.get("coalesced_unknown"))
        traces = [value for value in (previous_trace, latest_trace) if value]
        latest.__dict__["_execution_evidence"] = {
            **(latest_trace or previous_trace),
            "coalesced_trace_ids": [value.get("trace_id") for value in traces],
            "completed": complete,
            "coalesced_unknown": not complete,
            "counts": {kind: sum(value.get("counts", {}).get(kind, 0) for value in traces)
                       for kind in ("model_load", "model_warmup", "local_inference", "api_inference")},
            "events": [event for value in traces for event in value.get("events", [])],
        }
    confirmed = retain_confirmed_items(previous.items, latest.items, latest.source_items)
    latest.items = merge_items(previous.items, confirmed, latest.changed_regions)
    if latest.source_items is not None:
        latest.source_items = merge_items(previous.source_items or [], latest.source_items, latest.changed_regions)
    latest.retry_regions = list(dict.fromkeys([
        *(region for region in previous.retry_regions
          if not any(region.intersects(dirty) for dirty in latest.changed_regions)),
        *latest.retry_regions,
    ]))
    latest.is_full_frame = previous.is_full_frame
    latest.changed_regions = (
        [] if latest.is_full_frame else _combine_regions(
            previous.changed_regions, latest.changed_regions
        )
    )
    return latest


def _visible_change(item: object) -> bool:
    """判断一条渲染内容相对原文是否有可见变化。

    忽略首尾空白与内部连续空白：OCR 与模型都可能增减空格，
    那不算"翻译生效"。
    """
    text = getattr(item, "text", "") or ""
    source = getattr(item, "source_text", "") or ""
    if not source:
        # 没有原文可比（理论上不该出现）：保守认为发生了变化，
        # 避免把正常的翻译误报成"什么都没做"
        return True
    return _normalise_text(text) != _normalise_text(source)


def _normalise_text(text: str) -> str:
    """折叠空白，便于比较两段文字是否只是格式差异。"""
    return " ".join(text.split())


class _FpsCounter:
    """滑动窗口帧率统计。"""
    __slots__ = ("_ticks", "_window")
    def __init__(self, window: float = 2.0) -> None:
        self._ticks: list[float] = []
        self._window = float(window)

    def tick(self) -> None:
        """记录一次事件。"""
        now = time.perf_counter()
        self._ticks.append(now)
        cutoff = now - self._window
        if len(self._ticks) > 512:
            self._ticks = [stamp for stamp in self._ticks if stamp >= cutoff]

    def fps(self) -> float:
        """返回窗口内的平均帧率。"""
        if len(self._ticks) < 2:
            return 0.0
        span = self._ticks[-1] - self._ticks[0]
        if span <= 0:
            return 0.0
        return (len(self._ticks) - 1) / span

    def reset(self) -> None:
        """清空统计。"""
        self._ticks.clear()
