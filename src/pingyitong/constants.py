"""项目全局常量定义。

集中管理语言、模型、帧率、文件系统路径等跨模块共享的常量，
避免在各模块中散落硬编码字符串。
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------
# 应用标识
# --------------------------------------------------------------------------

APP_NAME: Final[str] = "屏译通"
APP_NAME_EN: Final[str] = "PingYiTong"
APP_SLOGAN: Final[str] = "基于本地大模型的桌面实时原位翻译系统"
APP_ORG: Final[str] = "PingYiTong Project"
APP_ID: Final[str] = "com.pingyitong.desktop.translator"

#: Windows 任务栏图标分组标识，必须在创建任何窗口前设置。
APP_USER_MODEL_ID: Final[str] = "com.pingyitong.desktop.translator.1"

#: 环境变量：覆盖应用数据目录，便于便携模式与测试隔离。
ENV_HOME: Final[str] = "PINGYITONG_HOME"

#: 环境变量：显式指定模型权重目录，优先级高于项目内与应用数据目录。
ENV_MODELS_DIR: Final[str] = "PINGYITONG_MODELS_DIR"

# --------------------------------------------------------------------------
# 支持语言
# --------------------------------------------------------------------------

#: 语言清单的**唯一来源**是 :mod:`pingyitong.languages`：
#: 那里维护了 142 种语言的代码、中文名、英文名、母语名与文字脚本，
#: 以及由 ``scripts/test_languages.py`` 实测得到的支持档位。
#:
#: 这里**不再另存一份语言表**。曾同时存在一份 8 种语言的 ``LANGUAGES``
#: 和一份 ``LANGUAGE_EN_NAMES``，两份表与实际可选项不一致——
#: 界面只给出 8 种，而模型实际能处理上百种。两份表就是两处会漂移的真相。
#:
#: 取用方式::
#:
#:     from pingyitong.languages import LANGUAGES, verification_status

#: 自动检测源语言的语言代码
AUTO_LANGUAGE: Final[str] = "auto"

#: 自动检测选项的界面文案
LANGUAGE_AUTO_LABEL: Final[str] = "自动检测"

# --------------------------------------------------------------------------
# 双模型定义（Spark-X2.5-1.7B / 4B）
# --------------------------------------------------------------------------

#: 极速模式模型标识
MODEL_FAST: Final[str] = "spark-x2.5-1.7b"
#: 高精度模式模型标识
MODEL_ACCURATE: Final[str] = "spark-x2.5-4b"

MODEL_ORDER: Final[tuple[str, ...]] = (MODEL_FAST, MODEL_ACCURATE)

MODEL_LABELS: Final[dict[str, str]] = {
    MODEL_FAST: "极速模式 (Spark-X2.5-1.7B)",
    MODEL_ACCURATE: "高精度模式 (Spark-X2.5-4B)",
}

MODEL_SHORT_LABELS: Final[dict[str, str]] = {
    MODEL_FAST: "极速 1.7B",
    MODEL_ACCURATE: "高精度 4B",
}

#: 模型文件名，与官方 Hugging Face 仓库发布的文件名保持一致。
#:
#: 来源：``XHToken/Spark-X2.5-1.7B-GGUF`` 与 ``XHToken/Spark-X2.5-4B-GGUF``
#: （Apache-2.0）。下载方式见 ``scripts/download_models.py``。
#:
#: 程序查找权重时对大小写不敏感，也支持"参数量特征匹配"，
#: 因此从 Ollama / LM Studio 下载的文件即使名字不同也能被识别，
#: 详见 :func:`pingyitong.utils.paths.resolve_model_path`。
MODEL_FILE_NAMES: Final[dict[str, str]] = {
    MODEL_FAST: "Spark-X2.5-1.7B-Q4_K_M.gguf",
    MODEL_ACCURATE: "Spark-X2.5-4B-Q4_K_M.gguf",
}

#: 模型权重的官方下载地址，用于界面提示与错误信息。
MODEL_REPOS: Final[dict[str, str]] = {
    MODEL_FAST: "XHToken/Spark-X2.5-1.7B-GGUF",
    MODEL_ACCURATE: "XHToken/Spark-X2.5-4B-GGUF",
}

#: 各模型 Q4_K_M 量化后的体积（字节），用于下载校验与内存估算提示。
MODEL_FILE_SIZES: Final[dict[str, int]] = {
    MODEL_FAST: 1_107_457_856,
    MODEL_ACCURATE: 2_600_224_352,
}

# --------------------------------------------------------------------------
# 调度模式（见《项目全栈开发文档》三、双模型动态调度策略）
# --------------------------------------------------------------------------

SCHEDULE_MODE_MANUAL: Final[str] = "manual"
SCHEDULE_MODE_AUTO: Final[str] = "auto"

SCHEDULE_MODE_LABELS: Final[dict[str, str]] = {
    SCHEDULE_MODE_MANUAL: "手动模式",
    SCHEDULE_MODE_AUTO: "自动调度",
}

#: 资源自适应阈值
CPU_HIGH_THRESHOLD: Final[float] = 70.0      # CPU 持续高于 70% 切 1.7B
RENDER_FPS_LOW_THRESHOLD: Final[float] = 10.0  # 渲染帧率低于 10fps（仅用于展示）
CPU_HIGH_SUSTAIN_SECONDS: Final[float] = 3.0   # 需持续多久才判定为高负载
GPU_FREE_MEMORY_MIN_MB: Final[int] = 2048      # GPU 显存空闲阈值

#: 端到端延迟阈值（秒）：超过则降级到 1.7B 以保证响应速度。
#:
#: 这是"用户是否在等"的直接度量，取代了早期用渲染帧率做降级依据的做法。
#: 渲染帧率在静态屏幕上天然趋近 0（没有新内容可渲染），用它降级会把
#: "屏幕没变"误判成"界面卡顿"，并与内容自适应升级形成死循环。
LATENCY_HIGH_THRESHOLD: Final[float] = 3.0

#: 内容自适应：文本长度阈值（字符数）
CONTENT_LONG_TEXT_CHARS: Final[int] = 80
#: 内容自适应：触发高精度模型的专业术语密度阈值
CONTENT_TERM_DENSITY_THRESHOLD: Final[float] = 0.15

#: 4B 模型空闲卸载超时（秒）
ACCURATE_MODEL_IDLE_TIMEOUT: Final[float] = 180.0

# --------------------------------------------------------------------------
# 性能与流水线参数
# --------------------------------------------------------------------------

DEFAULT_CAPTURE_FPS: Final[int] = 15
MIN_CAPTURE_FPS: Final[int] = 1
MAX_CAPTURE_FPS: Final[int] = 30

#: 流水线队列容量。达到容量时按"丢弃最旧帧"策略背压，
#: 确保翻译结果始终对应当前屏幕内容（见开发文档二·（六））。
FRAME_QUEUE_SIZE: Final[int] = 2
OCR_QUEUE_SIZE: Final[int] = 3
TRANSLATE_QUEUE_SIZE: Final[int] = 3
RESULT_QUEUE_SIZE: Final[int] = 3

#: 渲染轮询间隔（毫秒）
#:
#: 10ms（100Hz）让新译文尽快上屏：25ms 轮询时译文平均要多等约
#: 半个轮询周期才绘制，与"逐帧翻译"的低延迟目标相悖。空转 tick
#: 只检查队列是否为空，开销可忽略。
RENDER_POLL_INTERVAL_MS: Final[int] = 10

#: "屏幕内容无需翻译"提示的冷却时间（秒）。
#:
#: 屏幕上只要有一小块内容在变（光标闪烁、时钟走动），判断结果就会在
#: "有变化 / 无变化"之间来回翻转；若只在结论变化时才判断，提示会反复弹出
#: （实测 9 秒内弹了两次）。因此同一结论在该时间内只提示一次。
SKIP_NOTICE_COOLDOWN: Final[float] = 60.0

#: 画面差分参数
DIFF_PIXEL_THRESHOLD: Final[int] = 24      # 灰度差值阈值
DIFF_MIN_AREA_RATIO: Final[float] = 0.0004  # 变化面积占全屏比例下限
DIFF_BLOCK_SIZE: Final[int] = 16           # 差分网格块大小

#: OCR 文本块置信度下限
OCR_MIN_CONFIDENCE: Final[float] = 0.5
#: 小于该高度的文本块视为噪声丢弃
OCR_MIN_BOX_HEIGHT: Final[int] = 6
#: 文本长度上限（超出则截断，避免异常输入拖垮推理）
MAX_TEXT_CHARS: Final[int] = 2000

# --------------------------------------------------------------------------
# 缓存
# --------------------------------------------------------------------------

TRANSLATION_CACHE_MAX_ENTRIES: Final[int] = 4096
TRANSLATION_CACHE_TTL_SECONDS: Final[float] = 1800.0

# --------------------------------------------------------------------------
# 术语库
# --------------------------------------------------------------------------

#: 术语替换占位符。可读标记比私用区字符更容易被本地模型原样复制。
GLOSSARY_TOKEN_TEMPLATE: Final[str] = "[TERM_{}]"
GLOSSARY_TOKEN_PATTERN: Final[str] = r"\[TERM_(\d+)\]"

# --------------------------------------------------------------------------
# 翻译 Prompt 分隔符
# --------------------------------------------------------------------------

#: 批量翻译时用于分隔待翻译片段，要求模型原样保留。
SEGMENT_DELIMITER: Final[str] = "<<<SEG>>>"

# --------------------------------------------------------------------------
# 界面
# --------------------------------------------------------------------------

#: 悬浮控制条与遮罩层使用的强调色
ACCENT_COLOR: Final[str] = "#2E6BE6"
ACCENT_COLOR_HOVER: Final[str] = "#4A83F0"
SUCCESS_COLOR: Final[str] = "#2FA36B"
WARNING_COLOR: Final[str] = "#D98A1F"
ERROR_COLOR: Final[str] = "#D64545"

# --------------------------------------------------------------------------
# 资源文件名
# --------------------------------------------------------------------------

ICON_APP_PNG: Final[str] = "app_icon.png"
ICON_APP_ICO: Final[str] = "app_icon.ico"
ICON_TRAY_ACTIVE: Final[str] = "tray_active.png"
ICON_TRAY_IDLE: Final[str] = "tray_idle.png"
ICON_TRAY_PAUSED: Final[str] = "tray_paused.png"
