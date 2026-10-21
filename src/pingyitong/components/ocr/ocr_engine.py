"""文本检测与识别（OCR）模块。

开源组件：``RapidOCR``（RapidAI/RapidOCR，Apache-2.0 License）
    基于 PaddleOCR 导出的 ONNX 模型，无需安装深度学习框架，
    CPU 端单帧百毫秒级，支持 80 余种语言印刷体识别。

自研内容：桌面场景文本后处理、重复文本去重、文本行合并与段落分段
    （见 :mod:`pingyitong.services.text_preprocess`）。

坐标还原
    RapidOCR 返回的坐标相对**送入的图像**。若预处理阶段做了放大，
    需按放大倍数折算回原始尺度，再叠加该帧的 ``origin`` 得到物理坐标。
"""

from __future__ import annotations

import importlib.util
import threading
import time
import unicodedata
from collections import OrderedDict
from typing import Any, NamedTuple, Protocol

from ...languages import LANGUAGES
from ...utils.logger import get_logger
from ...utils.paths import asset_path
from ..capture.geometry import Rect
from ..image import downscale_boxes, downscale_for_detection, enhance_for_ocr
from .recognizer_profiles import (
    CHINESE,
    RecognizerProfile,
    profile_for_source,
    reverse_rtl_prediction,
    source_profile_key,
)
from .text_block import RecognizedFrame, TextBlock

logger = get_logger(__name__)


class OcrBackend(Protocol):
    """OCR 后端协议，便于替换为其他识别引擎。"""

    name: str

    def is_ready(self) -> bool:
        """后端是否已就绪可用。"""

    def recognize(self, image: Any) -> list[TextBlock]:
        """识别一幅图像（局部坐标，原点在图像左上角）。"""


# ----------------------------------------------------------------------
# RapidOCR 后端
# ----------------------------------------------------------------------


class RapidOcrBackend:
    """RapidOCR 后端封装。

    使用已验证的 ``rapidocr_onnxruntime`` API 与本地 ONNX 模型。
    不切换到可能隐式下载默认模型的新 API。
    """

    name = "RapidOCR"

    def __init__(self, threads: int = 0, profile: RecognizerProfile = CHINESE) -> None:
        """
        :param threads: onnxruntime 的 intra-op 线程数；``0`` 表示按
            物理核数自动取值（上限 8）。取值依据见
            :attr:`~pingyitong.config.schema.OcrConfig.threads`。
        """
        self._threads = max(0, int(threads))
        self.profile = profile
        self._engine: Any = None
        self._import_error: str = ""
        self._loaded = False
        self._load_lock = threading.Lock()
        # 裁剪级识别缓存：像素不变的文本行不再进识别模型。
        # 预热线程与 OCR 线程都会调 recognize()，必须加锁。
        self._crop_cache: dict[str, Any] = {}
        self._crop_cache_lock = threading.Lock()
        # text_rec 的输入宽度是可变配置；预热与 OCR 线程不能同时改写它。
        self._rec_lock = threading.Lock()
        self._decision_state = threading.local()

    @property
    def decision_events(self) -> list[dict]:
        return getattr(self._decision_state, "events", [])

    def is_ready(self) -> bool:
        """尝试加载引擎；未安装依赖时返回 ``False`` 并记录原因。"""
        with self._load_lock:
            return self._load_once()

    def _load_once(self) -> bool:
        if self._loaded:
            return self._engine is not None
        self._loaded = True

        try:
            self._engine = self._create_engine()
        except ImportError as exc:
            self._import_error = str(exc)
            logger.warning(
                "未检测到 RapidOCR，OCR 功能不可用。"
                "请执行 pip install rapidocr-onnxruntime 后重启。"
            )
        except Exception as exc:  # noqa: BLE001 - 模型加载失败原因多样，统一降级
            self._import_error = str(exc)
            logger.exception("RapidOCR 初始化失败")
        return self._engine is not None

    @property
    def error(self) -> str:
        """返回不可用的原因，供界面提示。"""
        return self._import_error

    @property
    def threads(self) -> int:
        """实际使用的线程数，供状态面板展示与排查。"""
        return resolve_threads(self._threads)

    #: 检测阶段的输入长边上限。检测耗时与输入像素数成正比，而字幕条
    #: 常宽达 1600px，字幕条降到 800 可减少检测耗时。完整文档不能沿用
    #: 这个上限：12–14px 小字会缩到不足 6px，产生漏框与半截框。
    DET_LONG_SIDE = 800
    #: 整屏/正文保留更高检测分辨率；薄字幕和局部变化仍走快速路径。
    DOCUMENT_DET_LONG_SIDE = 2000
    DOCUMENT_MIN_HEIGHT = 320

    #: 检测前给图像四周补的边距（像素）。DB 检测模型会把紧贴图像边缘
    #: 的长行漏检成半截——实测 1600×88 字幕行只返回行尾片段，而贴着
    #: 屏底/窗口边缘的真实字幕正是这种形态。补 24px 边距（复制边缘
    #: 像素）后 24 条基准台词全部完整识别，检测面积仅增约 6%，代价
    #: 可忽略；输出坐标会减去该值并夹回原图范围。
    EDGE_PAD = 24

    #: 送入识别模型的裁剪框外扩边距（像素）。检测框常比实际墨迹略窄，
    #: 行尾句号这类小构件会被切在框外（实测成对字幕行稳定丢句号）。
    #: 外扩 4px 后裁剪含住笔画，识别不再缺尾；仅影响裁剪范围，
    #: 输出坐标仍取原始检测框。
    CROP_PAD = 4

    def _create_engine(self) -> Any:
        """按可用包名创建引擎实例。

        显式传入 ``intra_op_num_threads``：RapidOCR 的默认线程配置在我们
        的负载上严重偏低效（实测慢 2.6 倍），原因见
        :attr:`~pingyitong.config.schema.OcrConfig.threads`。
        """
        threads = resolve_threads(self._threads)
        rec_path = self.profile.local_path()
        det_path = asset_path("ocr", "ch_PP-OCRv4_det_infer.onnx")
        cls_path = asset_path("ocr", "ch_ppocr_mobile_v2.0_cls_infer.onnx")
        for path in (det_path, cls_path):
            if not path.is_file():
                raise FileNotFoundError(f"缺少离线 OCR 模型：{path.name}（assets/ocr）")
        options = {
            "intra_op_num_threads": threads,
            "rec_model_path": str(rec_path),
            "det_model_path": str(det_path),
            "cls_model_path": str(cls_path),
            # 关闭文字方向分类器：桌面界面文字不会旋转，方向判别纯属浪费。
            # 实测整屏识别由 2.44s 降到 2.19s（约 10%）。
            "use_cls": False,
            # 检测模型输入尺寸限制。注意 ``limit_type='max'`` 时库**忽略**
            # 这里的 ``det_limit_side_len``（改按 <960/<1500/2000 分档），
            # 真正生效的是下面 recognize() 里的自行降采样；保留这两项只为
            # 保证库内部不把窄图放大（默认的 'min' 会把 900×120 放大到
            # 5520×736，OCR 从 ~80ms 变成 ~600ms）。
            "det_limit_side_len": 736,
            "det_limit_type": "max",
            # 关闭"极宽图自动上下补边"。库默认 ``width_height_ratio=8``：
            # 宽/高 > 8 时把图像补边到 ``max(宽/8, 30)*2`` 高，字幕条
            # 1600×84 因此被补成 1600×400，检测面积膨胀到 4.7 倍。
            # 实测关闭后 1600px 宽字幕条检测 78ms → 46ms，识别文本不变。
            "width_height_ratio": 10000,
            # 识别输入宽度保持库默认 [3,48,320]。曾固定为 48*28=1344 以
            # 消除动态 shape 抖动，但 ``resize_norm_img`` 把整批补边到
            # ``max(基准比, 批内最大比)``——基准比 28 意味着**每一批**都按
            # 最宽情形计算，普通批的识别成本直接翻倍。超宽行的补边问题
            # 由 ``_split_wide_crop`` 切段解决（批内最大比被压到 ≤16），
            # 剩余的首次 shape 编译开销（每种新宽度一次性 ~100ms）可接受。
        }

        # The installed 1.4.4 API reads the ONNX embedded dictionary. Newer
        # RapidOCR APIs may choose/download defaults; do not silently use them.
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR(**options)

    #: 裁剪缓存容量上限（按文本行计）。屏幕上的行数有限，512 行足够
    #: 覆盖典型桌面；超限清一半，避免长期运行内存无限增长。
    CROP_CACHE_SIZE = 512

    def recognize(self, image: Any) -> list[TextBlock]:
        """识别图像并转换为 :class:`TextBlock` 列表（局部坐标）。

        不走库的 ``__call__``：它把检测与识别绑在一起，检测必须在原尺寸
        图像上做。这里拆成"检测在小图上、识别框回原图裁剪"两步，是
        本模块最重要的性能优化（见 :attr:`DET_LONG_SIDE`）。

        识别阶段带**裁剪级缓存**：同一文本行像素不变时直接复用上次的
        (文本, 置信度)。实时翻译场景里屏幕大部分文字逐帧不变（只有
        字幕在滚动），不缓存的话每帧都为不变的行付全额识别成本，
        OCR 线程永远追不上采集帧率——实测静态屏幕上端到端延迟被推到
        数秒，而这些行的译文本来就命中翻译缓存，重复识别纯属浪费。
        """
        self._decision_state.events = []
        engine = self._engine
        if engine is None:
            self._decision_state.events.append({"stage": "ocr", "action": "reject",
                                               "reason": "backend_unavailable", "inputs": []})
            return []

        try:
            # 检测前给图像四周补一圈边距：DB 检测模型会把紧贴图像边缘
            # 的长行漏检成半截（实测 1600×88 字幕行只返回行尾片段，
            # 贴屏底/屏边的真实字幕即此形态）。补边后 24 条基准台词
            # 全部完整识别；坐标在输出前减回并夹回原图范围。
            padded, pad = _pad_edge(image, self.EDGE_PAD)
            boxes = self._detect(engine, padded)
            if boxes is None or len(boxes) == 0:
                self._decision_state.events.append({"stage": "ocr", "action": "observe",
                                                   "reason": "no_detected_boxes", "inputs": []})
                return []

            # 裁剪框四周外扩几个像素：DB 输出的四边形常比实际墨迹窄
            # （实测行尾句号被切在框外，识别成无句号的半截），外扩后
            # 裁剪含住笔画。输出坐标仍用原始框，避免渲染框变胖。
            crops = engine.get_crop_img_list(padded, _expand_recognition_boxes(boxes, self.CROP_PAD))
            if crops is None or len(crops) == 0:
                self._decision_state.events.append({"stage": "ocr", "action": "reject",
                                                   "reason": "no_valid_crops", "inputs": [],
                                                   "detected_boxes": len(boxes)})
                return []
            recognized = self._recognize_crops(engine, crops)
        except Exception:  # noqa: BLE001 - 单帧识别失败不应中断流水线
            self._decision_state.events.append({"stage": "ocr", "action": "reject",
                                               "reason": "backend_exception", "inputs": []})
            logger.exception("OCR 识别异常，已跳过本帧")
            return []

        height, width = image.shape[:2]
        blocks: list[TextBlock] = []
        for index, (box, item) in enumerate(zip(boxes, recognized, strict=False)):
            try:
                text, confidence = item[0], item[1]
            except (IndexError, TypeError):
                self._decision_state.events.append({"stage": "ocr", "action": "reject",
                                                   "reason": "invalid_recognizer_result",
                                                   "inputs": [], "detected_index": index})
                continue

            text = str(text).strip()
            rect = _polygon_to_rect(box)
            if rect is None:
                self._decision_state.events.append({"stage": "ocr", "action": "reject",
                                                   "reason": "invalid_detected_polygon",
                                                   "inputs": [], "detected_index": index})
                continue

            rect = _clamp_rect(
                Rect(rect.x - pad, rect.y - pad, rect.width, rect.height),
                width,
                height,
            )
            if not text or rect.is_empty:
                self._decision_state.events.append({
                    "stage": "ocr", "action": "filter",
                    "reason": "empty_recognizer_text" if not text else "outside_image",
                    "inputs": [{"index": index, "line_id": -1, "text": text,
                                "rect": list(rect.to_sequence()),
                                "confidence": float(confidence) if confidence is not None else 1.0}],
                })
                continue

            blocks.append(
                TextBlock(
                    text=text,
                    box=rect,
                    confidence=float(confidence) if confidence is not None else 1.0,
                    index=index,
                )
            )

        return blocks

    def _detect(self, engine: Any, image: Any) -> Any:
        """在降采样图上做文本检测，返回**原分辨率**坐标下的四边形。

        检测不到任何文本时退回原尺寸再检测一次：降采样可能漏掉很细的
        小字，这种情况宁可多花一次检测时间，也不能整帧识别为空——
        那会让用户以为翻译功能坏了。
        """
        import numpy

        # 对于密集小字，低分辨率仍能检出大标题，原来的“全空才重试”
        # 无法修复部分漏检。按画面形状区分文档与薄字幕，正文保留原尺寸
        # 到 2000px 上限；不先付一次必然丢小字的 800px 检测开销。
        long_side = self.DET_LONG_SIDE
        if image.shape[0] >= self.DOCUMENT_MIN_HEIGHT:
            long_side = max(long_side, self.DOCUMENT_DET_LONG_SIDE)
        small, scale = downscale_for_detection(image, long_side)
        boxes, _ = engine.text_det(small)
        if (boxes is None or len(boxes) == 0) and scale > 1.0:
            # 降采样后一个框都没检出来：退回原尺寸再试一次（比例即 1.0）
            boxes, _ = engine.text_det(image)
            if boxes is None or len(boxes) == 0:
                return boxes
            return numpy.asarray(engine.sorted_boxes(boxes), dtype="float32")

        if boxes is None or len(boxes) == 0:
            return boxes

        # ``sorted_boxes`` 返回的是列表（库内部实现如此），缩放前统一成数组
        ordered = numpy.asarray(engine.sorted_boxes(boxes), dtype="float32")
        if scale > 1.0:
            ordered = ordered * scale  # 小图坐标 → 原分辨率坐标
        return ordered

    def _recognize_crops(self, engine: Any, crops: list[Any]) -> list[Any]:
        """带裁剪级缓存的批量识别：命中缓存的行不再进识别模型。

        超宽裁剪块（整行代码、长句）会先按空白列切段再识别：
        ``text_rec`` 把一批裁剪块全部补边到本批**最大**长宽比，混入一个
        ratio≈60 的宽行会把同批普通块的耗时拖高 20 多倍（实测混批
        887ms，拆段后 59ms），整屏 29 行时端到端被推到 4 秒以上。

        :param engine: RapidOCR 实例（使用其 ``text_rec``）。
        :param crops: 待识别的文本行裁剪图列表。
        :return: 与 ``crops`` 顺序一一对应的 ``(文本, 置信度)`` 列表。
        """
        import hashlib

        results: list[Any] = [None] * len(crops)
        miss_indexes: list[int] = []
        miss_digests: list[str] = []

        for index, crop in enumerate(crops):
            digest = hashlib.blake2b(crop.tobytes(), digest_size=12).hexdigest()
            digest = f"{self.profile.key}:{crop.shape}:{crop.dtype}:{digest}"
            with self._crop_cache_lock:
                cached = self._crop_cache.get(digest)
            if cached is not None:
                results[index] = cached
            else:
                miss_indexes.append(index)
                miss_digests.append(digest)

        if miss_indexes:
            # 切段后统一送识别：段的长宽比都落在常规范围，批内补边
            # 不再被个别宽行放大。
            segment_groups: list[list[Any]] = []
            flat_segments: list[Any] = []
            for index in miss_indexes:
                crop = _prepare_recognition_crop(crops[index])
                # RTL decoding needs the complete visual line: splitting an
                # ASCII run inside Arabic and reversing each segment corrupts
                # the run order. Keep RTL lines whole, then decode once.
                parts = (
                    [_Segment(crop, True)]
                    if self.profile.rtl else _split_wide_crop(crop)
                )
                segment_groups.append(parts)
                flat_segments.extend(part.image for part in parts)

            with self._rec_lock:
                recognizer = engine.text_rec
                shape = getattr(recognizer, "rec_image_shape", None)
                # 少量字幕行固定到 20:1 的识别宽度：每种新宽度都让
                # ONNX Runtime 支付一次内存规划成本，实测约 150~220ms。
                # 多行保持库原来的动态宽度，避免把整屏批次无谓补宽。
                fixed_width = (
                    isinstance(shape, list)
                    and len(shape) >= 3
                    and len(flat_segments) <= 4
                )
                original_width = shape[2] if fixed_width else None
                try:
                    if fixed_width:
                        shape[2] = max(original_width, 960)
                    recognized, _ = recognizer(flat_segments, False)
                finally:
                    if fixed_width:
                        shape[2] = original_width

            position = 0
            for index, digest, parts in zip(miss_indexes, miss_digests, segment_groups, strict=True):
                segment_results = recognized[position : position + len(parts)]
                position += len(parts)
                if self.profile.rtl:
                    segment_results = [
                        (reverse_rtl_prediction(str(item[0])), item[1])
                        for item in segment_results
                    ]
                joined = _join_segment_results(segment_results, parts)
                results[index] = joined
                with self._crop_cache_lock:
                    self._crop_cache[digest] = joined
                    if len(self._crop_cache) > self.CROP_CACHE_SIZE:
                        # 超限保留最近写入的一半：简单且无需逐次维护链表
                        keep = list(self._crop_cache.items())[-self.CROP_CACHE_SIZE // 2 :]
                        self._crop_cache = dict(keep)

        return results


class NullOcrBackend:
    """占位后端：依赖缺失时使用，保证流水线可启动并给出明确提示。

    它会返回空结果而不是伪造识别内容，界面据此提示用户安装组件。
    """

    name = "未安装"

    def __init__(self, reason: str = "") -> None:
        self._reason = reason or "未安装 RapidOCR 组件"

    def is_ready(self) -> bool:
        return False

    @property
    def error(self) -> str:
        return self._reason

    def recognize(self, image: Any) -> list[TextBlock]:
        return []


# ----------------------------------------------------------------------
# 引擎
# ----------------------------------------------------------------------


#: 自动线程数的上限。实测 8 是本负载下的最优值：识别阶段要逐个处理几十个
#: 很小的文本块，线程再多会因同步开销反超收益（详见 OcrConfig.threads）。
_MAX_AUTO_THREADS = 8


def resolve_threads(configured: int) -> int:
    """把配置的线程数解析为实际使用的值。

    :param configured: 配置值；``0`` 表示自动。
    :return: 实际线程数。自动值取 ``min(8, 物理核数)``——
        上限 8 是因为本负载的推理输入很小，线程过多反而更慢；
        同时不超过物理核数，避免在核心少的机器上过度订阅。
    """
    if configured > 0:
        return configured

    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or 1
    except Exception:  # noqa: BLE001 - psutil 缺失时退化为逻辑核数
        import os

        cores = max(1, (os.cpu_count() or 2) // 2)

    return max(1, min(_MAX_AUTO_THREADS, int(cores)))


class OcrEngine:
    """OCR 引擎：串联图像预处理、识别与坐标还原。"""

    def __init__(
        self,
        upscale_factor: float = 1.0,
        enable_preprocess: bool = True,
        min_confidence: float = 0.5,
        min_box_height: int = 6,
        box_padding: int = 2,
        max_blocks_per_frame: int = 120,
        threads: int = 0,
        backend: OcrBackend | None = None,
        source_language: str = "auto",
        retain_uncertain: bool = True,
    ) -> None:
        self.upscale_factor = max(0.5, float(upscale_factor))
        self.enable_preprocess = bool(enable_preprocess)
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.min_box_height = max(1, int(min_box_height))
        self.retain_uncertain = bool(retain_uncertain)
        self.box_padding = max(0, int(box_padding))
        self.max_blocks_per_frame = max(1, int(max_blocks_per_frame))
        self._threads_config = max(0, int(threads))
        self.source_language = str(source_language)
        self._profile_key = source_profile_key(self.source_language)
        self._backend_generation = 0
        self._backend: OcrBackend | None = backend
        #: 后端懒初始化锁。预热线程（启动后 300ms）与流水线 OCR 线程
        #: 可能同时首次访问 :attr:`backend`——无锁时各自构造一次
        #: RapidOcrBackend，ONNX 会话初始化约 6 秒与双份内存全部翻倍，
        #: 其中一个实例被静默丢弃。
        self._backend_lock = threading.RLock()
        # Loading never holds the configuration lock: changing source in the
        # UI only changes state; ONNX construction stays in OCR/warm-up workers.
        self._backend_load_lock = threading.Lock()
        self._profile_backends: OrderedDict[str, OcrBackend] = OrderedDict()
        # Warm-up and OCR may run on separate threads. Audit state belongs to
        # the current call's thread, so warm-up cannot replace screen evidence.
        self._decision_state = threading.local()
        if backend is not None:
            self._profile_backends[self._profile_key] = backend
        self.MAX_RESIDENT_PROFILES = 3

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------

    def configure(
        self,
        upscale_factor: float | None = None,
        enable_preprocess: bool | None = None,
        min_confidence: float | None = None,
        min_box_height: int | None = None,
        box_padding: int | None = None,
        threads: int | None = None,
        source_language: str | None = None,
        retain_uncertain: bool | None = None,
    ) -> None:
        """运行时更新识别参数。

        ``threads`` 变更需要重建推理会话才能生效：若已创建后端且线程数
        确实变了，就丢弃后端让它按新参数重建（重建耗时约 0.3 秒，
        由下一次识别承担）。
        """
        with self._backend_lock:
            if retain_uncertain is not None:
                self.retain_uncertain = bool(retain_uncertain)
            if upscale_factor is not None:
                self.upscale_factor = max(0.5, float(upscale_factor))
            if enable_preprocess is not None:
                self.enable_preprocess = bool(enable_preprocess)
            if min_confidence is not None:
                self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
            if min_box_height is not None:
                self.min_box_height = max(1, int(min_box_height))
            if box_padding is not None:
                self.box_padding = max(0, int(box_padding))

            if threads is not None and max(0, int(threads)) != self._threads_config:
                self._threads_config = max(0, int(threads))
                self._profile_backends.clear()
                self._backend = None
                self._backend_generation += 1
            if source_language is not None:
                self.source_language = str(source_language)
                profile_key = source_profile_key(self.source_language)
                if profile_key != self._profile_key:
                    self._profile_key = profile_key
                    self._backend_generation += 1
                    self._backend = self._profile_backends.get(profile_key)
                    if self._backend is not None:
                        self._profile_backends.move_to_end(profile_key)

    @property
    def threads(self) -> int:
        """实际使用的 onnxruntime 线程数（已把"自动"解析为具体值）。

        注意这**不是**配置值：配置里 ``0`` 表示自动，会自动解析为
        ``min(8, 物理核数)``。状态面板与日志应显示这个解析后的值，
        否则排查性能问题时容易误以为线程设置没生效。
        """
        return resolve_threads(self._threads_config)

    @property
    def configured_threads(self) -> int:
        """配置中的原始线程数（``0`` 表示自动）。"""
        return self._threads_config

    @property
    def backend(self) -> OcrBackend:
        """Lazily load the selected offline profile, avoiding stale load commits."""
        while True:
            with self._backend_lock:
                if self._backend is not None:
                    return self._backend
            with self._backend_load_lock:
                with self._backend_lock:
                    if self._backend is not None:
                        return self._backend
                    source = self.source_language
                    profile_key = self._profile_key
                    threads = self._threads_config
                try:
                    candidate = RapidOcrBackend(
                        threads=threads, profile=profile_for_source(source)
                    )
                    loaded = candidate if candidate.is_ready() else NullOcrBackend(candidate.error)
                except ValueError as exc:
                    loaded = NullOcrBackend(str(exc))
                with self._backend_lock:
                    if threads == self._threads_config:
                        self._profile_backends[profile_key] = loaded
                        self._profile_backends.move_to_end(profile_key)
                        while len(self._profile_backends) > self.MAX_RESIDENT_PROFILES:
                            self._profile_backends.popitem(last=False)
                    # A source can change away and back while this model loads.
                    # Commit by active profile and thread config, not an old
                    # generation: the loaded alphabet is still exactly valid.
                    if threads == self._threads_config and profile_key == self._profile_key:
                        self._backend = loaded
                        return loaded

    @property
    def backend_name(self) -> str:
        """状态展示不得触发同步模型加载。"""
        with self._backend_lock:
            return self._backend.name if self._backend is not None else RapidOcrBackend.name

    def is_ready(self) -> bool:
        """OCR 是否可用。"""
        return self.backend.is_ready()

    def preflight_error(self) -> str:
        """Check offline resources without constructing an ONNX session."""
        try:
            profile_for_source(self.source_language).local_path()
            for filename in ("ch_PP-OCRv4_det_infer.onnx", "ch_ppocr_mobile_v2.0_cls_infer.onnx"):
                if not asset_path("ocr", filename).is_file():
                    raise FileNotFoundError(f"缺少离线 OCR 模型：{filename}（assets/ocr）")
            if importlib.util.find_spec("rapidocr_onnxruntime") is None:
                return "未安装 rapidocr-onnxruntime，无法识别屏幕文字"
        except (ValueError, FileNotFoundError) as exc:
            return self._source_error(str(exc))
        return ""

    def warm_up(self) -> bool:
        """预热检测与常见识别 batch shape。

        只用空白图只能预热检测网络，识别网络直到用户第一帧才真正运行。
        而识别网络对 batch=1/2/3/4 的首次执行各有一次 100~200ms 的内存规划
        成本。用四张合成字幕图把常见 shape 一次编译完，避免这笔成本落到
        用户看到的首帧延迟里。

        :return: 预热是否成功。
        """
        backend = self.backend
        if not backend.is_ready():
            return False

        try:
            import cv2
            import numpy as np

            for line_count in (1, 2, 3, 4):
                image = np.full((line_count * 64 + 20, 1600, 3), 255, dtype=np.uint8)
                for index in range(line_count):
                    cv2.putText(
                        image,
                        f"Warm up line {index + 1}",
                        (20, 52 + index * 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                backend.recognize(image)
        except Exception:  # noqa: BLE001 - 预热失败不影响后续使用
            logger.debug("OCR 预热失败（不影响后续使用）")
            return False
        with self._backend_lock:
            return backend is self._backend

    def error_message(self) -> str:
        """不可用时的原因说明。"""
        with self._backend_lock:
            backend = self._backend
        if backend is None:
            return self.preflight_error()
        error = getattr(backend, "error", "")
        return self._source_error(str(error)) if error else ""

    def _source_error(self, error: str) -> str:
        info = LANGUAGES.get(self.source_language)
        source = f"{info.name}（{info.script}）" if info else self.source_language
        return f"源语言 {source}：{error}"

    # ------------------------------------------------------------------
    # 识别
    # ------------------------------------------------------------------

    @property
    def decision_events(self) -> list[dict]:
        events = getattr(self._decision_state, "events", None)
        if events is None:
            events = []
            self._decision_state.events = events
        return events

    def recognize_frame(self, frame: Any) -> RecognizedFrame:
        """识别一整帧（按变化区域分块识别）。

        :param frame: :class:`~pingyitong.components.capture.CaptureFrame`。
        :return: :class:`RecognizedFrame`，文本块坐标为全局物理坐标。
        """
        started = time.perf_counter()
        origin = frame.origin
        is_full = frame.is_full_frame
        collected: list[TextBlock] = []
        events: list[dict] = []

        for region in frame.effective_regions():
            patch = frame.crop(region)
            if patch is None or patch.size == 0:
                continue
            local_blocks = self.recognize_image(patch)
            # 局部坐标 -> 该帧的全局物理坐标
            offset_x = origin.x + region.left
            offset_y = origin.y + region.top
            for event in self.decision_events:
                translated = dict(event)
                translated["region"] = [offset_x, offset_y, region.width, region.height]
                translated["inputs"] = [
                    {**item, "rect": [item["rect"][0] + offset_x,
                                      item["rect"][1] + offset_y, *item["rect"][2:]]}
                    for item in event.get("inputs", [])
                ]
                events.append(translated)
            for block in local_blocks:
                collected.append(
                    block.with_text(block.text, block.box.translate(offset_x, offset_y))
                )

        # 区域识别可能产生跨区域重复（差分框有重叠），在此去重。
        before_deduplication = collected
        collected = self._deduplicate(collected)
        retained_ids = {id(block) for block in collected}
        for removed in before_deduplication:
            if id(removed) not in retained_ids:
                events.append({"stage": "ocr", "action": "deduplicate",
                               "reason": "same_text_overlapping_regions",
                               "inputs": [{"index": removed.index, "line_id": removed.line_id,
                                           "text": removed.text,
                                           "rect": list(removed.box.to_sequence()),
                                           "confidence": removed.confidence}]})
        collected.sort(key=lambda b: (b.box.top, b.box.left))
        for index, block in enumerate(collected):
            block.index = index

        return RecognizedFrame(
            seq=frame.seq,
            origin=origin,
            blocks=collected,
            elapsed=time.perf_counter() - started,
            is_full_frame=is_full,
            monitor_index=frame.monitor_index,
            diagnostics={
                "ocr_recognized": len(collected),
                "ocr_uncertain": sum(block.ocr_uncertain for block in collected),
            },
            decision_events=events,
        )

    def recognize_image(self, image: Any) -> list[TextBlock]:
        """识别单幅图像，返回局部坐标的文本块。

        内部完成：预处理 -> 识别 -> 坐标还原 -> 过滤。
        """
        self._decision_state.events = []
        if image is None or getattr(image, "size", 0) == 0:
            return []

        backend = self.backend
        if not backend.is_ready():
            self.decision_events.append({"stage": "ocr", "action": "reject",
                                         "reason": "backend_unavailable", "inputs": []})
            return []

        processed = image
        scale = 1.0
        if self.enable_preprocess:
            scale = self.upscale_factor
            processed = _prepare_ocr_image(image, scale)
            if _low_contrast_range(image) is not None:
                self.decision_events.append({
                    "stage": "ocr", "action": "preprocess",
                    "reason": "low_contrast_preprocessing", "inputs": [],
                    "scale": scale,
                })

        blocks = backend.recognize(processed)
        for event in getattr(backend, "decision_events", []):
            restored_event = dict(event)
            restored_event["inputs"] = [
                {**item, "rect": list(downscale_boxes([tuple(item["rect"])], scale)[0])}
                for item in event.get("inputs", [])
            ]
            self.decision_events.append(restored_event)
        if not blocks:
            return []

        if abs(scale - 1.0) > 1e-3:
            blocks = _rescale_blocks(blocks, scale)

        return self._filter(blocks)

    # ------------------------------------------------------------------
    # 过滤与去重
    # ------------------------------------------------------------------

    def _filter(self, blocks: list[TextBlock]) -> list[TextBlock]:
        """保留可见候选；低置信和小字只标记，不凭阈值删除。"""
        accepted: list[TextBlock] = []
        for block in blocks:
            if block.is_empty or block.box.is_empty:
                self._record_filter(block, "filter", "empty_text" if block.is_empty else "empty_box")
                continue
            uncertain = (
                block.confidence < self.min_confidence
                or block.box.height < self.min_box_height
            )
            if uncertain and not self.retain_uncertain:
                self._record_filter(block, "filter", "low_confidence" if block.confidence < self.min_confidence else "small_box")
                continue
            block.ocr_uncertain = uncertain
            if uncertain:
                self._record_filter(block, "retain", "low_confidence" if block.confidence < self.min_confidence else "small_box")
            accepted.append(block)

        if len(accepted) > self.max_blocks_per_frame:
            # 这是下游的分批预算，不应永久丢弃整屏后半部分或低一点的
            # 可信文字。翻译调度自行分批，本层不能以“保护延迟”截断页面。
            logger.debug(
                "单帧可信文本 %s 块超过处理预算 %s，已全量保留供分批处理",
                len(accepted), self.max_blocks_per_frame,
            )

        if self.box_padding:
            for block in accepted:
                block.box = block.box.expand(self.box_padding)

        return accepted

    def _record_filter(self, block: TextBlock, action: str, reason: str) -> None:
        self.decision_events.append({
            "stage": "ocr", "action": action, "reason": reason,
            "inputs": [{"index": block.index, "line_id": block.line_id,
                        "text": block.text, "rect": list(block.box.to_sequence()),
                        "confidence": block.confidence}],
        })

    @staticmethod
    def _deduplicate(blocks: list[TextBlock]) -> list[TextBlock]:
        """去除重复文本块（同文本且包围盒高度重叠）。"""
        accepted: list[TextBlock] = []
        for block in blocks:
            duplicate = False
            for existing in accepted:
                if existing.text != block.text:
                    continue
                overlap = existing.box.intersection(block.box)
                smaller = min(existing.box.area, block.box.area) or 1
                if overlap.area / smaller > 0.6:
                    # 保留置信度更高的那个
                    if block.confidence > existing.confidence:
                        accepted.remove(existing)
                        accepted.append(block)
                    duplicate = True
                    break
            if not duplicate:
                accepted.append(block)
        return accepted


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------


def _prepare_ocr_image(image: Any, scale: float) -> Any:
    """已有清晰笔画不再锐化；低对比画面才做 CLAHE 与边缘增强。"""
    import numpy

    sample = image[::4, ::4]
    gray = sample.mean(axis=2) if sample.ndim == 3 else sample
    low, high = numpy.percentile(gray, (1, 99))
    if high - low < 96:
        return _stretch_low_contrast(enhance_for_ocr(image, upscale=scale))
    # 默认的统一锐化会改变 12px 的细笔画，实测精确率从 195/200
    # 降到 154/200。高对比文字保持原始像素，用户显式缩放仍然生效。
    if abs(scale - 1.0) <= 1e-3:
        return image
    import cv2

    height, width = image.shape[:2]
    return cv2.resize(
        image, (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


def _low_contrast_range(image: Any) -> tuple[float, float] | None:
    """A bounded pixel statistic, never a guessed character/word repair."""
    if image is None or not hasattr(image, "shape") or image.ndim != 3:
        return None
    import cv2
    import numpy

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low, high = numpy.percentile(gray, (0.01, 99.99))
    return (float(low), float(high)) if 2 <= high - low < 96 else None


def _stretch_low_contrast(image: Any) -> Any:
    """Recover faint ink after CLAHE without changing already clear glyphs.

    CLAHE alone left a real 20-level faint row in an 18-level range and the
    detector cut off Japanese line tails while reporting high confidence.
    Stretch only a narrow measured luminance range; uniform/one-level noise
    and clear pages are left alone. No punctuation or lexical completion.
    """
    levels = _low_contrast_range(image)
    if levels is None:
        return image
    import cv2
    import numpy

    low, high = levels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = numpy.clip(
        (gray.astype(numpy.float32) - low) * (255.0 / (high - low)), 0, 255,
    ).astype(numpy.uint8)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def _prepare_recognition_crop(crop: Any) -> Any:
    """细字首次识别用三次插值放大，缓存命中不重复处理像素。"""
    if crop.shape[0] > 24:
        return crop
    import cv2

    # 识别器内部直接线性放到48px会破坏细笔画；真实12px基准
    # 三次插值可从195/200提升到198/200，14–18px不损失完整行。
    return cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)


def _polygon_to_rect(polygon: Any) -> Rect | None:
    """把四边形顶点转换为轴对齐包围盒。"""
    try:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return None

    if not xs or not ys:
        return None

    return Rect.from_ltrb(round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)))


def _pad_edge(image: Any, pad: int) -> tuple[Any, int]:
    """把图像四周复制边缘像素补宽 ``pad`` 像素，返回 ``(图像, 实际边距)``。

    复制边缘而非填黑：浅色界面上填黑会在文字与边框之间造出一圈高
    对比黑带，反而可能被检测模型当成文本行。
    """
    if pad <= 0 or image is None or not hasattr(image, "shape") or len(image.shape) < 2:
        return image, 0
    height, width = image.shape[:2]
    # 图像本身比边距还小时补出来只有噪声，跳过。
    if height <= pad or width <= pad:
        return image, 0

    import numpy

    # 只补 H/W 两轴；标量 pad_width 会连通道轴一起补，破坏图像形状。
    pad_width = [(pad, pad), (pad, pad)] + [(0, 0)] * (image.ndim - 2)
    padded = numpy.pad(image, pad_width, mode="edge")
    return padded, pad


def _clamp_rect(rect: Rect, width: int, height: int) -> Rect:
    """把矩形夹回 ``[0, width] × [0, height]`` 范围内（非负且不越界）。"""
    left = max(0, min(rect.x, width))
    top = max(0, min(rect.y, height))
    right = max(left, min(rect.x + rect.width, width))
    bottom = max(top, min(rect.y + rect.height, height))
    return Rect(left, top, right - left, bottom - top)


def _expand_boxes(boxes: Any, pad: int) -> Any:
    """把检测四边形向外扩 ``pad`` 像素（仅用于裁剪输入）。

    DB 输出的框比实际墨迹略窄，行尾句号会被切在裁剪之外，识别成
    缺尾的文本。扩框以框中心为基准沿 x/y 各向外推 ``pad``，裁剪
    稍大不影响识别（识别输入本来就归一化到固定高）。
    """
    if pad <= 0 or boxes is None:
        return boxes

    import numpy

    grown_list = []
    for box in numpy.asarray(boxes, dtype="float32"):
        points = box.reshape(-1, 2)
        x0, y0 = points[:, 0].min(), points[:, 1].min()
        x1, y1 = points[:, 0].max(), points[:, 1].max()
        rect = numpy.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype="float32"
        )
        center = rect.mean(axis=0)
        offset = rect - center
        # 按比例放大等价于外扩：宽/高各自至少 2*pad。
        scale_x = (x1 - x0 + 2 * pad) / max(x1 - x0, 1e-3)
        scale_y = (y1 - y0 + 2 * pad) / max(y1 - y0, 1e-3)
        grown = center + offset * numpy.array([scale_x, scale_y], dtype="float32")
        grown_list.append(grown)
    return numpy.stack(grown_list).astype("float32")


def _expand_recognition_boxes(boxes: Any, pad: int) -> Any:
    """小字使用较小裁剪边距，避免把邻行混入识别与过度缩小字形。"""
    if pad <= 0 or boxes is None:
        return boxes

    import numpy

    expanded = []
    for box in numpy.asarray(boxes, dtype="float32"):
        points = box.reshape(-1, 2)
        height = float(points[:, 1].max() - points[:, 1].min())
        # 固定 4px 让 12px 字体裁剪膨胀近一倍，真实基准 200 行仅
        # 148 行正确；1–2px 恢复到 195+ 行。字幕大字仍保留原来的 4px。
        local_pad = min(pad, max(1, round(height / 8.0)))
        expanded.append(_expand_boxes(numpy.asarray([box]), local_pad)[0])
    return numpy.asarray(expanded, dtype="float32")


def _rescale_blocks(blocks: list[TextBlock], scale: float) -> list[TextBlock]:
    """把放大图上得到的包围盒还原到原始尺度。"""
    boxes = downscale_boxes([b.box.to_sequence() for b in blocks], scale)
    restored: list[TextBlock] = []
    for block, (x, y, w, h) in zip(blocks, boxes, strict=True):
        restored.append(block.with_text(block.text, Rect(x, y, w, h)))
    return restored


# ----------------------------------------------------------------------
# 超宽裁剪块切分
# ----------------------------------------------------------------------

#: 识别输入允许的最大长宽比。超过它的裁剪块在送入 ``text_rec`` 前切段。
#: ``text_rec`` 把整批补边到本批最大长宽比，一个 ratio≈60 的宽行混进
#: 普通 batch，会把同批 5 块从 36ms 拖到 887ms（实机整屏 29 行时
#: OCR 达 4.3s 的根因）。
_MAX_REC_RATIO = 20.0

#: 切段的目标长宽比：与 ``rec_img_shape`` 的基准比（1344/48 = 28）对齐，
#: 段落在常规批内不再触发额外补边。
_SEGMENT_RATIO = 16.0


def _column_ink(crop: Any) -> Any:
    """按列统计"文字像素"密度（0~1，按行高归一），用于找字符间的空白列。

    文字与背景哪个亮哪个暗随主题变化，且文字永远是少数派——
    取与**中位数灰度**（背景色）偏差超过阈值的像素即可，无需关心极性。
    """
    import numpy

    gray = crop.astype(numpy.float32).mean(axis=2)
    background = numpy.median(gray)
    ink = (numpy.abs(gray - background) > 48.0).sum(axis=0).astype(numpy.float32)
    # 平滑 3 列，避免单个零宽列干扰 argmin；再按行高归一到 0~1
    kernel = numpy.ones(3, dtype=numpy.float32) / 3.0
    smoothed = numpy.convolve(ink, kernel, mode="same")
    return smoothed / max(1.0, float(crop.shape[0]))


#: 段间需要补空格的字符集：仅 ASCII 词字符与常见标点。中文不属于
#: 任何集合，因此中文段拼接不会插入空格。
_ASCII_TAIL = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.;:!?%)}]")
_ASCII_HEAD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789(${[\"'")


#: 切段结果：段图像 + 该段**左边界切点**是否落在真正的空白列上。
#: 空白切点处两段原本就是分开的词，拼接时按需补空格；
#: 硬切点（切穿了文字）必须直接相连，否则 "code" 会变成 "cod e"、
#: "识别" 会变成 "识 别"。
class _Segment(NamedTuple):
    image: Any
    from_space: bool


def _split_wide_crop(crop: Any) -> list[_Segment]:
    """把长宽比超限的裁剪块按空白列切成若干段。

    切点优先取目标位置邻域内"文字像素为零"的连续空白区中心——落在
    字符间隙上，不切字形；邻域内找不到空白列（密集长串）才硬切，
    极端情况下最多损伤切点处的一个字符。

    :return: 段列表（顺序与原块一致）；不需切分时返回原块单元素列表。
    """
    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        return [_Segment(crop, True)]
    ratio = width / float(height)
    if ratio <= _MAX_REC_RATIO:
        return [_Segment(crop, True)]

    count = int(-(-ratio // _SEGMENT_RATIO))  # ceil
    ideal = width / count
    window = max(8, int(ideal * 0.25))
    ink = _column_ink(crop)

    cuts = [0]
    from_space = [True]
    for step in range(1, count):
        target = int(round(step * ideal))
        low, high = max(1, target - window), min(width - 1, target + window)
        if high <= low:
            cuts.append(target)
            from_space.append(False)
            continue
        min_space_width = max(6, _SPACE_RUN_RATIO * height)
        column = _pick_cut_column(ink[low:high], target - low, min_space_width)
        cuts.append(low + column)
        # 切点必须落在**足够宽**的空白段上才算词间空格：字母 kerning
        # 间隙也是"空白列"，但只有几个像素宽，在那里切等价于切穿单词。
        run = _blank_run_width(ink, low + column)
        from_space.append(run >= min_space_width)
    cuts.append(width)

    segments: list[_Segment] = []
    for index, (start, end) in enumerate(zip(cuts, cuts[1:], strict=False)):
        if end > start:
            segments.append(_Segment(crop[:, start:end], from_space[index]))
    return segments or [_Segment(crop, True)]


#: 空白切点的判定阈值：该列的文字像素密度（按行归一）低于它即视为空白列。
_SPACE_INK_EPS = 0.02

#: 判定"词间空格"的最小空白段宽度（占行高比例）。比例字体的字母间隙
#: 只有 1~4px，而空格约 0.25em——按行高 18% 分界，才能区分"切在词间"
#: 与"切穿了单词"（实测仅按 ink≈0 判定会把 kerning 间隙误判为空格，
#: 拼出 "th e" 这类断裂词）。
_SPACE_RUN_RATIO = 0.18


def _blank_run_width(ink: Any, column: int) -> int:
    """返回 ``column`` 所在空白段的宽度（含端点）。"""
    start = column
    while start > 0 and ink[start - 1] <= _SPACE_INK_EPS:
        start -= 1
    end = column
    total = len(ink)
    while end + 1 < total and ink[end + 1] <= _SPACE_INK_EPS:
        end += 1
    return end - start + 1


def _pick_cut_column(ink_window: Any, offset: int, min_space_width: float) -> int:
    """在窗口内选切点列。

    优先选择足够宽的词间空白。距离目标更近的字母间隙也可能是空白列，
    在那里切段会让识别模型把半个单词读错。没有词间空白时才选择
    最近的较窄空白段，最后退回 ink 最小处。
    """
    import numpy

    threshold = _SPACE_INK_EPS
    blank = ink_window <= threshold
    if not blank.any():
        return int(numpy.argmin(ink_window))

    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(blank):
        if not blank[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(blank) and blank[index + 1]:
            index += 1
        runs.append((start, index))
        index += 1

    word_spaces = [run for run in runs if run[1] - run[0] + 1 >= min_space_width]
    choices = word_spaces or runs
    start, end = min(choices, key=lambda run: abs((run[0] + run[1]) // 2 - offset))
    return (start + end) // 2


def _join_segment_results(segment_results: list[Any], segments: list[_Segment]) -> Any:
    """把切段后的识别结果拼接为单条 ``(文本, 置信度)``。

    相邻两段之间的切点若是空白列且两侧都是 ASCII 词字符，补一个空格
    （英文词间隙）；硬切点或中文边界直接相连。
    置信度取各段最小值——任何一段不可靠，整行就不可靠。
    """
    text_parts: list[str] = []
    confidence = 1.0
    for item, segment in zip(segment_results, segments, strict=False):
        try:
            part_text, part_score = str(item[0]).strip(), float(item[1])
        except (IndexError, TypeError, ValueError):
            continue
        if not part_text:
            continue
        if (
            text_parts
            and segment.from_space
            and (
                (text_parts[-1][-1] in _ASCII_TAIL and part_text[0] in _ASCII_HEAD)
                or _space_separated_word_boundary(text_parts[-1][-1], part_text[0])
            )
        ):
            text_parts.append(" ")
        text_parts.append(part_text)
        confidence = min(confidence, part_score)
    return ("".join(text_parts), confidence)


def _space_separated_word_boundary(left: str, right: str) -> bool:
    """Preserve a measured word gap beyond ASCII (e.g. Cyrillic/Hangul).

    Han and kana normally have no word spaces; their interglyph gaps must not
    become new spaces. Combining marks are word characters in Indic scripts.
    Hard cuts never call this rule because ``from_space`` is false.
    """
    def word_char(char: str) -> bool:
        value = ord(char)
        no_word_space = (0x3400 <= value <= 0x9FFF or 0x3040 <= value <= 0x30FF
                         or 0xFF65 <= value <= 0xFF9F)
        return not no_word_space and unicodedata.category(char)[0] in {"L", "M", "N"}
    return word_char(left) and word_char(right)


def create_ocr_engine(config: Any, source_language: str | None = None) -> OcrEngine:
    """依据 OCR 配置节创建引擎。"""
    return OcrEngine(
        upscale_factor=getattr(config, "upscale_factor", 1.0),
        enable_preprocess=getattr(config, "enable_preprocess", True),
        min_confidence=getattr(config, "min_confidence", 0.5),
        min_box_height=getattr(config, "min_box_height", 6),
        retain_uncertain=getattr(config, "retain_uncertain", True),
        box_padding=getattr(config, "box_padding", 2),
        max_blocks_per_frame=getattr(config, "max_blocks_per_frame", 120),
        threads=getattr(config, "threads", 0),
        source_language=(source_language if source_language is not None else getattr(config, "source_language", "auto")),
    )
