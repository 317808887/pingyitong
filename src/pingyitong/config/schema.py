"""配置数据模型。

所有配置项以不可变风格的 dataclass 描述，通过 :meth:`to_dict` /
:func:`from_dict` 与 JSON 互转。新增配置项只需在此定义字段并给出默认值，
老版本配置文件缺失的字段会自动回落到默认值，实现向前兼容。
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

from ..constants import (
    ACCURATE_MODEL_IDLE_TIMEOUT,
    CONTENT_LONG_TEXT_CHARS,
    CPU_HIGH_SUSTAIN_SECONDS,
    CPU_HIGH_THRESHOLD,
    DEFAULT_CAPTURE_FPS,
    DIFF_BLOCK_SIZE,
    DIFF_MIN_AREA_RATIO,
    DIFF_PIXEL_THRESHOLD,
    GPU_FREE_MEMORY_MIN_MB,
    MODEL_ACCURATE,
    MODEL_FAST,
    MODEL_FILE_NAMES,
    OCR_MIN_BOX_HEIGHT,
    OCR_MIN_CONFIDENCE,
    RENDER_FPS_LOW_THRESHOLD,
    SCHEDULE_MODE_MANUAL,
    TRANSLATION_CACHE_MAX_ENTRIES,
    TRANSLATION_CACHE_TTL_SECONDS,
)

T = TypeVar("T")

# --------------------------------------------------------------------------
# 屏幕捕获
# --------------------------------------------------------------------------


@dataclass
class CaptureConfig:
    """桌面实时捕获参数。"""

    #: 捕获帧率，默认 15fps（见产品功能规划二·（一））
    fps: int = DEFAULT_CAPTURE_FPS
    #: 轻量画面监视间隔；稳定画面不重复执行 OCR / 模型。
    monitor_interval_ms: int = 50
    #: 捕获模式：``fullscreen`` 全域；``region`` 仅用户框选区域
    mode: str = "fullscreen"
    #: 区域模式下的矩形，逻辑坐标 ``[x, y, width, height]``；``None`` 表示全屏
    region: list[int] | None = None
    #: 参与翻译的显示器序号（``mss`` 的 monitor 索引，0 为全部显示器合成的虚拟屏）；
    #: 空列表表示所有显示器
    monitors: list[int] = field(default_factory=list)
    #: 是否启用画面差分（仅处理变化区域）
    enable_diff: bool = True
    diff_pixel_threshold: int = DIFF_PIXEL_THRESHOLD
    diff_min_area_ratio: float = DIFF_MIN_AREA_RATIO
    diff_block_size: int = DIFF_BLOCK_SIZE
    #: 是否根据系统负载动态调整帧率
    adaptive_fps: bool = True
    #: 捕获区域上下左右内缩像素，规避屏幕边缘控件
    edge_margin: int = 0


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------


@dataclass
class OcrConfig:
    """文字检测与识别参数。"""

    min_confidence: float = OCR_MIN_CONFIDENCE
    min_box_height: int = OCR_MIN_BOX_HEIGHT
    #: 保留低置信/小字候选并标记疑问，不用阈值静默丢弃。
    retain_uncertain: bool = True
    #: 文本检测框向外扩展像素，避免紧贴文字导致背景遮罩过窄
    box_padding: int = 2
    #: 是否对图像做小字号增强等预处理
    enable_preprocess: bool = True
    #: 放大倍数，针对小字号界面文字
    upscale_factor: float = 1.0
    #: 单帧最大文本块数量，超出部分丢弃，避免极端场景拖垮推理
    # 兼容既有配置的密集画面提示阈值；超过时不再永久丢弃可信文字块。
    max_blocks_per_frame: int = 120
    #: 文本行合并的垂直重叠比例阈值（原创后处理算法参数）
    line_merge_overlap_ratio: float = 0.55
    #: 归入同一文本行所允许的最大**水平**间隙，单位为字高倍数。
    #:
    #: 这是个关键参数，用于区分"同一栏内的相邻文字"与"并排的不同窗口"。
    #: 桌面截图里常常并排开着多个窗口/侧栏，它们在同一水平带上各有文字；
    #: 若只按垂直重叠分行，这些毫不相干的元素会被当成同一行，进而合并成
    #: 一个跨窗口的巨型段落，译文彻底错乱（实测把侧栏任务行、测试窗口标签、
    #: 悬浮条状态合并成一段 2167 像素宽的文本，渲染后完全错位）。
    #:
    #: 取值依据（2560×1600 桌面实测）：
    #:
    #: - 跨窗口间隙：2.3–2.6 倍字高（侧栏右边界到相邻窗口左边界）
    #: - 同栏内"标签 … 快捷键提示"的间隙：可达 3.4 倍字高
    #:
    #: **两种分布是重叠的**，所以不能靠间隙大小把它们分开；此外全局 x 投影
    #: 也没有空隙（不同行的文字覆盖了不同区间），按投影分栏同样不可行。
    #: 因此这里取一个偏保守的阈值：**只合并真正紧邻的文字**。
    #:
    #: 代价是"标签 + 远端数值"这类同栏元素会各自成段。这是刻意接受的取舍：
    #: 对屏幕翻译而言，**位置正确比翻译语境重要**——拆开只会让每段译文各自
    #: 落在自己的位置上，而错误合并会让整片译文错位。
    line_max_horizontal_gap_ratio: float = 1.5
    #: 同行文本块合并的水平间距阈值（相对字高倍数）
    same_line_gap_ratio: float = 1.6
    #: 段落划分的垂直间距阈值（相对行高倍数）
    paragraph_gap_ratio: float = 1.7
    #: onnxruntime 的 intra-op 线程数；``0`` 表示自动。
    #:
    #: **这是 OCR 性能的关键参数，不要随意改动。** 实测（24 核机器、
    #: 整屏 2560×1600、74 个文本块）：
    #:
    #: ================  ==========
    #: 线程数             耗时
    #: ================  ==========
    #: 默认（不指定）      5.89 s
    #: 1                 8.08 s
    #: 2                 4.73 s
    #: 4                 3.00 s
    #: **8**             **2.25 s**
    #: 12                2.28 s
    #: 16                2.62 s
    #: ================  ==========
    #:
    #: 原因：识别阶段要逐个处理几十个文本块，每块的推理输入都很小
    #: （约 100×32 像素），线程过多时**线程同步开销反超计算收益**。
    #: 自动值取 ``min(8, 物理核数)``。
    threads: int = 0


# --------------------------------------------------------------------------
# 文本预处理
# --------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    """OCR 后处理与文本清洗参数（原创算法，见原创性声明二·（五））。"""

    #: 合并被换行截断的句子
    merge_broken_lines: bool = True
    #: 修复行尾连字符断词（英文）
    repair_hyphenation: bool = True
    #: 过滤重复文本块
    deduplicate: bool = True
    #: 过滤纯符号、纯数字噪声
    filter_noise: bool = True
    #: 忽略纯数字内容（如行号、价格），避免无意义翻译
    skip_pure_numeric: bool = True
    #: 源语言与目标语言相同时跳过翻译
    skip_same_language: bool = True


# --------------------------------------------------------------------------
# 大模型
# --------------------------------------------------------------------------


@dataclass
class SingleModelConfig:
    """单个本地模型的推理参数。"""

    enabled: bool = True
    file_name: str = ""
    #: 上下文窗口
    n_ctx: int = 2048
    #: 推理线程数，0 表示按 CPU 核心数自动
    #:
    #: **不要让它占满全部核心**：OCR 与推理是流水线上相邻的两级，
    #: 推理把 CPU 吃满时 OCR 会明显变慢（实测整屏 OCR 从 1.0 秒涨到 5 秒以上）。
    #: 自动值取 ``max(4, 物理核数 // 3)``，给 OCR 与界面留出余量。
    n_threads: int = 0
    #: 卸载到 GPU 的层数，0 表示纯 CPU 推理
    n_gpu_layers: int = 0
    #: 批处理大小
    n_batch: int = 256
    #: 翻译是确定性任务；真机语义样本在 0.0 下更稳定。已有用户配置仍保留原值。
    temperature: float = 0.0
    top_p: float = 0.9
    #: 单次生成的最大 token 数
    max_tokens: int = 512
    #: 是否常驻内存（加载后不因空闲而卸载）
    keep_resident: bool = False
    #: 是否在程序启动时预加载。
    #:
    #: 极速模型默认预加载：它是常驻模型，且首次翻译就很可能用到，
    #: 预加载可以把"第一次翻译的等待"提前到启动阶段。
    #: 高精度模型默认不预加载：它占用约 2.8GB 内存，
    #: 在 8GB 内存的机器上与系统争抢内存反而拖慢整体，
    #: 因此保持"首次使用时加载 + 空闲自动卸载"的策略。
    preload: bool = False


# --------------------------------------------------------------------------
# 在线 AI 模型（OpenAI 兼容接口）
# --------------------------------------------------------------------------


@dataclass
class ApiConfig:
    """外部 AI 模型接入配置（OpenAI ``/chat/completions`` 兼容协议）。

    任何兼容该协议的服务（DeepSeek、智谱、Kimi、通义、OpenAI、
    本机 Ollama 等）填三样即可用：接口地址、API Key、模型名。
    启用后翻译请求优先路由到云端模型，失败自动回退本地模型。

    Key 明文保存在本机配置文件中（见 :mod:`.settings` 的路径），
    与浏览器保存密码同等安全级别；也可留空 Key（如本机 Ollama）。
    """

    #: 是否把翻译路由到在线模型（失败自动回退本地，不丢译文）
    enabled: bool = False
    #: 接口根地址，例如 ``https://api.deepseek.com/v1``；
    #: 程序在其后拼接 ``/chat/completions``
    base_url: str = ""
    #: API Key；留空表示服务不需要鉴权（如本机 Ollama）
    api_key: str = ""
    #: 模型名称，例如 ``deepseek-chat``
    model_name: str = ""
    #: 采样温度。翻译任务低温，与本地模型 :class:`SingleModelConfig` 一致
    temperature: float = 0.2
    #: 单次请求输出 token 上限（引擎侧兜底默认）。翻译服务实际按源文本
    #: 动态计算输出上限：本地模式受模型上限钳制；在线模式以动态估计
    #: （2.5×源 token + 分隔符记账）为主导、8192 硬上限防跑飞，
    #: 避免长段落译文被默认值截断（本字段未在界面暴露）。
    max_tokens: int = 768
    #: 请求超时（秒）。在线模型典型响应 0.5–2 秒，超时视为本次失败
    timeout_seconds: float = 10.0
    #: 在线模式下的批量片段数。云端模型上下文长（≥8k），可一次携带更多片段，
    #: 减少请求往返次数；本地批量大小（``model.batch_size``）不适用
    batch_size: int = 10
    #: 批量请求的并发数。整帧多批时并发发送，整帧延迟约降为 1/N；
    #: 过大易触发服务商限流，默认 4
    max_concurrency: int = 4


@dataclass
class ModelConfig:
    """双模型配置。"""

    fast: SingleModelConfig = field(
        default_factory=lambda: SingleModelConfig(
            file_name=MODEL_FILE_NAMES[MODEL_FAST],
            keep_resident=True,
            preload=True,
            n_ctx=2048,
            max_tokens=384,
        )
    )
    accurate: SingleModelConfig = field(
        default_factory=lambda: SingleModelConfig(
            file_name=MODEL_FILE_NAMES[MODEL_ACCURATE],
            keep_resident=False,
            preload=False,
            # 翻译任务用不到超长上下文：一次批量请求（系统提示 + 16 段）
            # 实测约 1000 token，2048 足够。更长的上下文只会白白扩大
            # KV 缓存的显存占用（用户明确要求 max_context=2048）。
            n_ctx=2048,
            max_tokens=768,
        )
    )
    #: 批量翻译时一次送入的最大片段数
    batch_size: int = 6
    #: 推理后端选择
    #:
    #: ``True``：优先使用 ``llama-server`` 外部进程。这是 Spark-X2.5 唯一
    #: 可行的方式——它的 ``spark2_5`` 架构只有 XHToken 维护的 llama.cpp
    #: 分支支持，而上游 llama.cpp（即 llama-cpp-python 捆绑的那个）
    #: 不认识该架构。详见 ``components/llm/backend.py``。
    #:
    #: ``False``：只使用 ``llama-cpp-python`` 进程内推理，适用于架构受
    #: 上游支持的其它 GGUF 模型。Spark-X2.5 用它会加载失败。
    prefer_server_backend: bool = True
    #: 是否启用模型的思考（推理）过程。
    #:
    #: Spark-X2.5 是推理模型，默认会先输出思考过程再给答案。桌面实时翻译
    #: 需要即时响应，因此默认**关闭**——实测同一句话在 1.7B 上：
    #: 关闭思考约 0.1 秒，开启思考约 0.6 秒（慢 6 倍）。
    #:
    #: 开启后措辞可能更贴切（例如 "Save As" 关闭时译"保存为"、
    #: 开启时译"另存为"，后者才是 Windows 菜单的标准译法），
    #: 适合对翻译质量要求高于响应速度的场景。
    enable_thinking: bool = False
    #: 推理缓存
    enable_cache: bool = True
    cache_max_entries: int = TRANSLATION_CACHE_MAX_ENTRIES
    cache_ttl_seconds: float = TRANSLATION_CACHE_TTL_SECONDS

    def for_model(self, model_id: str) -> SingleModelConfig:
        """按模型标识取对应配置。"""
        return self.fast if model_id != MODEL_ACCURATE else self.accurate


# --------------------------------------------------------------------------
# 调度
# --------------------------------------------------------------------------


@dataclass
class ScheduleConfig:
    """双模型动态调度参数（见开发文档三）。

    默认值以**延迟优先**为准，依据是真机实测：

    ====================  ==================  ==========
    模型                  单帧 45 段翻译       相对速度
    ====================  ==================  ==========
    极速 Spark-X2.5-1.7B  0.30 s              1×
    高精度 Spark-X2.5-4B  5.47 s              **慢 18 倍**
    ====================  ==================  ==========

    对实时翻译工具而言，18 倍的差距直接决定可用性——4B 每帧要 5 秒以上，
    用户看到的是译文迟迟不出现。因此默认固定使用 1.7B，"内容自适应升级到
    4B"默认关闭：它换来的是更好的措辞，代价是十倍的延迟，
    应由用户按场景自行权衡。
    """

    #: ``manual`` 手动模式 / ``auto`` 自动调度。
    #: 默认手动 + 极速模型 = 开箱即用的最低延迟。
    mode: str = SCHEDULE_MODE_MANUAL
    #: 手动模式下固定使用的模型；自动调度下作为初始模型
    active_model: str = MODEL_FAST
    #: 资源自适应开关
    resource_adaptive: bool = True
    cpu_high_threshold: float = CPU_HIGH_THRESHOLD
    cpu_high_sustain_seconds: float = CPU_HIGH_SUSTAIN_SECONDS
    render_fps_low_threshold: float = RENDER_FPS_LOW_THRESHOLD
    gpu_free_memory_min_mb: int = GPU_FREE_MEMORY_MIN_MB
    #: 内容自适应开关：长句/专业术语密集时**升级**到高精度模型。
    #:
    #: **默认关闭**，因为它是延迟的主要来源：实测同一帧 45 段文本，
    #: 4B 需要 5.47 秒而 1.7B 只要 0.30 秒（慢 18 倍）。实时翻译场景下
    #: 每帧多等 5 秒通常得不偿失，需要更贴切措辞时再手动开启。
    #: 关闭后仍有资源自适应（降级保护），只是不会为内容主动升级。
    content_adaptive: bool = False
    long_text_chars: int = CONTENT_LONG_TEXT_CHARS
    #: 4B 模型空闲卸载超时（秒）
    accurate_idle_timeout: float = ACCURATE_MODEL_IDLE_TIMEOUT
    #: 启动时自动预加载模型。
    #:
    #: 默认开启：把"第一次翻译的等待"提前到启动阶段，用户点开始翻译就能用。
    #: 具体加载哪些模型由各模型的 ``preload`` 字段决定
    #: （极速模型默认加载、高精度模型默认不加载，见
    #: :class:`SingleModelConfig`）。
    #: 权重文件缺失时会自动跳过，不会让启动卡住。
    preload_on_start: bool = True


# --------------------------------------------------------------------------
# 翻译
# --------------------------------------------------------------------------


@dataclass
class TranslationConfig:
    """翻译行为配置。"""

    #: 源语言，``auto`` 表示自动检测
    source_language: str = "auto"
    #: 目标语言
    target_language: str = "zh"
    #: 自动检测不可靠时用的兜底源语言
    fallback_source_language: str = "en"
    #: 是否在译文后附加原文（对照模式）
    keep_source_text: bool = False
    #: 本地隐私模式禁止在线翻译、预连接和联网预取。
    local_only: bool = True
    #: 生成完整译文后重新理解源文并审查，失败保留原文。
    semantic_review: bool = True
    #: 审查策略（问答 #9 分级可靠性）：
    #: ``always``——全部译文都走模型二重审查（最高延迟）；
    #: ``risk``——只有含否定、条件/时间、数字或高危动作词的段落才进
    #: 模型二审，低风险段落由确定性守卫（结构/语言/数字/术语完整性）
    #: 把关后直接发布。小字 UI 文案大多低风险，这是延迟与"整批扣留
    #: 导致只显示原文"两类问题的共同出口。
    review_policy: str = "always"
    #: 完整译文始终保留；原位另用经过审查的精简措辞。
    compact_display: bool = True
    #: auto / subtitle / dense / document / dynamic
    scene_mode: str = "auto"
    #: 语义精简：在不改变语义的前提下尽量缩短译文长度。
    #:
    #: 只压缩措辞（去掉源文中隐含的客套、冗余代词、重复限定），
    #: **不得省略专业术语、学术词汇、数字、单位与代码标识符**——
    #: 那些一旦丢失，译文就从"更短"变成"更错"了。
    concise_output: bool = False
    #: 术语库
    glossary_enabled: bool = True
    #: 空值兼容旧配置：旧用户已有 CSV 路径时继续使用自定义库。
    glossary_mode: str = ""
    #: 预装库领域；自定义 CSV 模式下保留上次选择，切回时恢复。
    glossary_domain: str = "software_ai"
    glossary_path: str = ""
    glossary_case_sensitive: bool = False


# --------------------------------------------------------------------------
# 渲染样式
# --------------------------------------------------------------------------


@dataclass
class StyleConfig:
    """译文渲染样式（见产品功能规划三·（三））。"""

    #: ``cover`` 完全覆盖原文 / ``overlay`` 半透明叠加
    mode: str = "cover"
    font_family: str = ""
    #: ``auto`` 依据原文包围盒自动匹配字号 / ``fixed`` 使用固定字号
    font_size_mode: str = "auto"
    font_size_fixed: int = 14
    #: 自动模式下的字号缩放系数
    font_scale: float = 1.0
    min_font_size: int = 12
    max_font_size: int = 44
    bold: bool = False
    text_color: str = "#FFFFFF"
    bg_color: str = "#101418"
    bg_opacity: float = 0.62
    bg_radius: int = 4
    padding_x: int = 3
    padding_y: int = 1
    shadow: bool = True
    shadow_color: str = "#000000"
    shadow_offset: int = 1
    #: 行距倍数
    line_spacing: float = 1.08
    #: 单个文本块最多渲染行数，避免溢出遮挡
    max_lines: int = 6
    #: 译文相对原文包围盒的整体像素偏移
    offset_x: int = 0
    offset_y: int = 0


# --------------------------------------------------------------------------
# 热键
# --------------------------------------------------------------------------


@dataclass
class HotkeyConfig:
    """全局热键绑定。

    键名取自 Qt 的 ``QKeySequence`` 可解析字符串，便于在界面上直接录入。
    """

    enabled: bool = True
    toggle_translation: str = "Ctrl+Alt+T"
    switch_model: str = "Ctrl+Alt+M"
    copy_result: str = "Ctrl+Alt+C"
    toggle_overlay: str = "Ctrl+Alt+H"


# --------------------------------------------------------------------------
# 历史记录
# --------------------------------------------------------------------------


@dataclass
class HistoryConfig:
    """翻译历史回溯（见产品功能规划三·（五））。"""

    enabled: bool = True
    #: 最大保留条数，0 表示不限制
    max_entries: int = 20000
    #: 是否记录屏幕识别原文（隐私敏感，用户可关闭）
    store_source_text: bool = True


# --------------------------------------------------------------------------
# 界面
# --------------------------------------------------------------------------


@dataclass
class UiConfig:
    """界面外观与交互。"""

    #: ``dark`` / ``light`` / ``system``
    theme: str = "dark"
    #: 悬浮控制条是否显示
    floating_bar_visible: bool = True
    #: 悬浮控制条位置（逻辑坐标）
    floating_bar_pos: list[int] | None = None
    #: 启动后最小化到系统托盘
    start_minimized: bool = True
    #: 开机自启
    autostart: bool = False
    #: 是否已完成首次引导
    first_run_done: bool = False
    #: 翻译层显示开关（用户临时一键隐藏后的状态）
    overlay_visible: bool = True
    #: 界面显示语言
    language: str = "zh"
    #: 是否遵循系统"减少动态效果"设置
    respect_reduced_motion: bool = True
    #: 手动开启减少动态效果（覆盖系统设置）
    reduced_motion: bool = False
    #: 通知（Toast）开关
    notifications: bool = True


# --------------------------------------------------------------------------
# 根配置
# --------------------------------------------------------------------------


@dataclass
class PrefetchConfig:
    """联网预翻译（P1 骨架，方案见 docs/联网预翻译技术方案.md v1.1）。

    **开关只管网络抓取与预热；缓存直读常开**（§12-D1：读缓存不产生
    流量，且旧译文失效由 TTL/版本机制独立保证）。所有阈值不写死。
    P0 探针已回填：fuzzy_window_n=4（±3 在 15% 丢行率下不够）、
    fuzzy_ratio=0.80（命中率 99.0%/错配 0.0%）。
    """

    enabled: bool = False
    wifi_only: bool = True
    #: 可达性探测
    probe_interval_s: float = 10.0
    probe_timeout_ms: int = 2000
    #: 空闲判定
    idle_cpu_threshold: float = 30.0
    idle_input_seconds: float = 5.0
    #: 模型空闲窗：距最近一次推理超过该秒数才允许预翻译请求出队。
    #: 压测教训：句边界让出不够——预翻译请求与实时请求在时间上重叠时，
    #: 实时 p95 劣化 +88%（连续负载最坏情况），远超 5% 门槛。
    #: 只有模型完全空闲一段窗口才出队，实时突发期间预翻译完全让出。
    idle_model_seconds: float = 3.0
    #: 配额（触顶 → QUOTA_PAUSED，次日窗口重置）
    requests_per_minute: int = 12
    max_page_bytes: int = 2 * 1024 * 1024
    daily_traffic_mb: int = 100
    prefetch_next_max: int = 5
    coldstart_probe_per_site_day: int = 20
    #: 预翻译 worker 固定 1（接缝 B：实时任务句边界抢占）
    translate_workers: int = 1
    model_queue_depth_max: int = 0
    #: 持久缓存
    cache_max_mb: float = 64.0
    subtitle_ttl_days: float = 30.0
    site_ttl_days: float = 7.0
    #: fuzzy（P0 标定值）
    fuzzy_ratio: float = 0.80
    fuzzy_window_n: int = 4
    fuzzy_sequence_check: int = 2
    #: 每源健康
    health_failure_n: int = 3
    cooldown_s: float = 120.0
    #: 盈亏门控（仅网页下一章）：a* = C / (ΔL·V + C)
    econ_C_s: float = 1.0
    econ_delta_L_s: float = 2.0
    econ_V: float = 1.0
    #: 指标保留期
    metrics_retention_days: float = 30.0
    #: 播放位置预热目标缓冲秒数
    buffer_target_s: float = 60.0
    #: 连播自动换轨（默认提示，不自动）
    auto_advance_track: bool = False
    #: 隐私同意（首次开启弹窗后置 true，拒绝则开关回落）
    consent_granted: bool = False
    #: 字幕目录监视：这些目录里出现新字幕文件（SRT/VTT/ASS）时
    #: 自动解析并注册为字幕轨（PotPlayer/VLC 等播放器的字幕目录）
    subtitle_dirs: list[str] = field(default_factory=list)


@dataclass
class AppConfig:
    """应用总配置。"""

    #: 配置文件版本，用于未来迁移
    config_version: int = 1
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)
    #: 最近一次运行时使用的模型，仅作展示
    last_active_model: str = MODEL_ACCURATE


# --------------------------------------------------------------------------
# 序列化
# --------------------------------------------------------------------------


def to_dict(obj: Any) -> Any:
    """将配置对象递归转换为可 JSON 序列化的字典。"""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


def _coerce(value: Any, target_type: Any) -> Any:
    """将 JSON 中的值尽量转换为声明的类型。

    配置文件可能被手工编辑，类型可能写成 ``"15"`` 之类的字符串，
    此处做宽松转换，转换失败时返回 ``None`` 由调用方回落默认值。
    """
    if value is None:
        return None

    origin = get_origin(target_type)
    if origin is list or origin is tuple:
        args = get_args(target_type)
        if not isinstance(value, (list, tuple)):
            return None
        if not args:
            return list(value)
        inner = args[0]
        converted = []
        for item in value:
            coerced = _coerce(item, inner)
            if coerced is None and (item is not None or not _allows_none(inner)):
                return None
            converted.append(coerced)
        return converted

    if origin is not None:
        # Optional[...] / Union[...]
        candidates = [a for a in get_args(target_type) if a is not type(None)]
        for candidate in candidates:
            coerced = _coerce(value, candidate)
            if coerced is not None:
                return coerced
        return None

    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    if target_type is int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            try:
                return int(value)
            except (ValueError, OverflowError):
                return None
        if isinstance(value, str):
            try:
                return int(float(value.strip()))
            except (ValueError, OverflowError):
                return None
        return None

    if target_type is float:
        if isinstance(value, (int, float, str)) and not isinstance(value, bool):
            try:
                number = float(value)
                return number if math.isfinite(number) else None
            except (ValueError, OverflowError):
                return None
        return None

    if target_type is str:
        return value if isinstance(value, str) else str(value)

    return value


def _allows_none(target_type: Any) -> bool:
    """只有可空字段才能保留 JSON null，其他字段使用默认值。"""
    return target_type is Any or target_type is type(None) or type(None) in get_args(target_type)


def from_dict(cls: type[T], data: Any, path: str = "", *, _defaults: T | None = None) -> T:
    """从字典构造配置对象。

    未知键被忽略；缺失或类型非法的键回落到 dataclass 默认值，
    因此升级版本后旧配置文件仍可正常加载。
    """
    defaults = _defaults if _defaults is not None else cls()
    if not isinstance(data, dict):
        return defaults

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {f.name: getattr(defaults, f.name) for f in fields(cls)}

    for f in fields(cls):
        if f.name not in data:
            continue
        raw = data[f.name]
        target = hints.get(f.name, Any)

        if dataclasses.is_dataclass(target):
            if isinstance(raw, dict):
                # 子节的默认工厂可能覆盖类默认值（例如 fast/accurate 的模型文件名）。
                # 递归携带该字段的默认实例，缺失与非法值均保持对应模型的默认值。
                kwargs[f.name] = from_dict(
                    target, raw, f"{path}.{f.name}", _defaults=getattr(defaults, f.name)
                )
        else:
            coerced = _coerce(raw, target)
            if coerced is not None or (raw is None and _allows_none(target)):
                kwargs[f.name] = coerced

    return cls(**kwargs)


def merge_into(base: AppConfig, patch: dict[str, Any]) -> AppConfig:
    """将字典补丁**递归**合并到现有配置并返回新对象。

    用于按节更新配置（例如仅更新 ``ui.theme`` 一个字段）。

    **为什么必须递归**：合并只做一层时，形如
    ``{"model": {"accurate": {"n_ctx": 2048}}}`` 的两级补丁会把整个
    ``accurate`` 子节**替换**成 ``{"n_ctx": 2048}``，其余字段（模型文件名、
    采样参数…）全部回落到类默认值——实测发生过：模型文件名被清空导致
    高精度模型"消失"。设置面板的所有开关都发这类局部补丁，
    因此这里必须逐层深入，只覆盖补丁里出现的字段。
    """
    current = to_dict(base)
    _deep_merge(current, patch)
    return from_dict(AppConfig, current)


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> None:
    """就地递归合并 ``patch`` 到 ``target``；字典逐层深入，其余直接覆盖。"""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
