"""设计令牌（Design Tokens）。

本文件是界面的**唯一视觉真理来源**：颜色、间距、圆角、字号、动效参数
全部在此定义，所有界面代码只能引用令牌，不得内联魔法数字。

设计规范来源
------------

**mozilla/OpenDesign**（已归档，MPL-2.0 + 设计作品 CC0-1.0）
    该项目主张"把开源原则带入创意设计"，其 README 的核心要求是
    "Once you've completed a project please include the source files!"
    据此在本项目中落实四项原则：

    1. **交付设计源文件**——本令牌文件与 ``docs/UI设计规范.md`` 即为
       设计源文件本身，可被非开发者直接审阅与修改，而不是一张无法编辑的截图；
    2. **开放可复用**——设计令牌以 CC0 精神公开，许可与出处写入文档；
    3. **隐私与信任优先**——界面必须如实呈现"全离线"这一核心承诺，
       不设置任何诱导性文案（对应"隐私不出端"的产品定位）；
    4. **无障碍优先**——正文对比度不低于 WCAG AA（4.5:1），
       关键信息不依赖颜色单独传达。

**emilkowalski/skills**（设计工程规范）
    动效相关的令牌（时长、缓动曲线、缩放值）严格取自该规范所定义的
    数值，不自行发明近似值——规范明确要求"Never invent a curve
    because it looks familiar"。具体取值见 :data:`MOTION`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# ==========================================================================
# 颜色
# ==========================================================================


@dataclass(frozen=True)
class ColorScheme:
    """一套完整配色方案。

    命名遵循语义而非外观（``surface`` 而不是 ``gray800``），
    这样切换主题时界面代码无需改动。
    """

    #: 窗口背景
    background: str
    #: 卡片 / 面板背景
    surface: str
    #: 抬升表面（弹层、悬浮条）
    surface_raised: str
    #: 悬停态表面
    surface_hover: str
    #: 按下态表面
    surface_active: str
    #: 分隔线与控件边框
    border: str
    #: 强调色边框（聚焦态）
    border_focus: str

    #: 主文字（正文）
    text_primary: str
    #: 次要文字（说明、单位）
    text_secondary: str
    #: 弱化文字（占位、禁用）
    text_muted: str
    #: 反色文字（用于强调色背景之上）
    text_inverse: str

    #: 品牌强调色
    accent: str
    #: 强调色悬停
    accent_hover: str
    #: 强调色按下
    accent_active: str
    #: 强调色的低透明度底衬
    accent_subtle: str
    #: 可直接作为文字使用的强调色（深色背景下需要更亮的色值才能满足对比度）
    accent_text: str

    #: 成功
    success: str
    #: 警告
    warning: str
    #: 错误 / 危险操作
    danger: str
    #: 危险色低透明度底衬
    danger_subtle: str

    #: 翻译层默认遮罩（深色界面）
    overlay_mask_dark: str
    #: 翻译层默认遮罩（浅色界面）
    overlay_mask_light: str


DARK_SCHEME: Final[ColorScheme] = ColorScheme(
    background="#0F1115",
    surface="#171A21",
    surface_raised="#1E222B",
    surface_hover="#22262F",
    surface_active="#282D38",
    border="#2A2F3A",
    border_focus="#3D6FD8",
    text_primary="#E8EAED",
    text_secondary="#A6AEBD",
    text_muted="#7C8598",
    text_inverse="#FFFFFF",
    accent="#2E6BE6",
    accent_hover="#4A83F0",
    accent_active="#275BC4",
    accent_subtle="#1B2C4D",
    # 深色背景上 #2E6BE6 作为文字对比度不足，使用提亮版本
    accent_text="#7FA8FF",
    success="#3FBF7F",
    warning="#E0A03C",
    danger="#E5645F",
    danger_subtle="#3A2224",
    overlay_mask_dark="#101418",
    overlay_mask_light="#F2F4F7",
)

LIGHT_SCHEME: Final[ColorScheme] = ColorScheme(
    background="#F5F6F8",
    surface="#FFFFFF",
    surface_raised="#FFFFFF",
    surface_hover="#F0F2F5",
    surface_active="#E7EAEF",
    border="#E1E4EA",
    border_focus="#2E6BE6",
    text_primary="#1B1F27",
    text_secondary="#5A6473",
    text_muted="#8A93A3",
    text_inverse="#FFFFFF",
    accent="#2E6BE6",
    accent_hover="#1F58CC",
    accent_active="#1A4AAE",
    accent_subtle="#E8F0FE",
    accent_text="#1F58CC",
    success="#1E9E63",
    warning="#B87A1A",
    danger="#CE3B36",
    danger_subtle="#FDECEB",
    overlay_mask_dark="#101418",
    overlay_mask_light="#FFFFFF",
)


# ==========================================================================
# 间距、圆角、描边
# ==========================================================================

#: 4px 基准网格
SPACE: Final[dict[str, int]] = {
    "none": 0,
    "xxs": 2,
    "xs": 4,
    "sm": 8,
    "md": 12,
    "lg": 16,
    "xl": 20,
    "xxl": 24,
    "xxxl": 32,
}

RADIUS: Final[dict[str, int]] = {
    "sm": 6,
    "md": 8,
    "lg": 12,
    "xl": 16,
    "pill": 999,
}

BORDER_WIDTH: Final[dict[str, int]] = {
    "none": 0,
    "thin": 1,
    "focus": 2,
}


# ==========================================================================
# 字体
# ==========================================================================

#: 字体族优先级。中文界面优先使用系统内置的各种黑体，
#: 避免因缺少字体而回退到点阵宋体，破坏界面观感。
FONT_FAMILIES: Final[tuple[str, ...]] = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "PingFang SC",
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "Segoe UI",
    "Arial",
)

#: 等宽字体，用于路径、快捷键等技术信息展示
MONO_FONT_FAMILIES: Final[tuple[str, ...]] = (
    "Cascadia Mono",
    "Consolas",
    "JetBrains Mono",
    "Courier New",
)

FONT_SIZE: Final[dict[str, int]] = {
    "caption": 11,
    "small": 12,
    "body": 13,
    "subtitle": 15,
    "title": 18,
    "display": 22,
}

FONT_WEIGHT: Final[dict[str, int]] = {
    "regular": 400,
    "medium": 500,
    "semibold": 600,
}

LINE_HEIGHT: Final[dict[str, float]] = {
    "tight": 1.25,
    "normal": 1.5,
    "relaxed": 1.7,
}


# ==========================================================================
# 动效令牌
# ==========================================================================

#: 缓动曲线控制点，与 CSS ``cubic-bezier(x1, y1, x2, y2)`` 一一对应。
#:
#: 规范强调内置的 ``ease-out`` 曲线"太弱"，必须使用这些更强的自定义曲线，
#: 否则动效会显得犹豫、不果断。
EASING: Final[dict[str, tuple[float, float, float, float]]] = {
    # 强 ease-out：元素进入 / 退出。起步快，立刻给出响应感
    "out": (0.23, 1.0, 0.32, 1.0),
    # 强 ease-in-out：屏幕内的位移与形变
    "in_out": (0.77, 0.0, 0.175, 1.0),
    # iOS 抽屉曲线（取自 Ionic Framework），用于大面积滑动面板
    "drawer": (0.32, 0.72, 0.0, 1.0),
}

#: 时长（毫秒）。规范硬性要求：界面动效一律不超过 300ms。
DURATION: Final[dict[str, int]] = {
    # 瞬时：键盘触发与高频操作用。规范规定"100+ 次/天 的操作永不加动画"
    "instant": 0,
    # 按钮按下反馈
    "press": 160,
    # 工具提示
    "tooltip": 125,
    # 下拉、弹出层、选择器
    "popover": 200,
    # 模态对话框
    "modal": 250,
    # 抽屉 / 大面板
    "drawer": 500,
    # 通知（Toast）——规范刻意比其他界面动效慢，以营造从容感
    "toast": 400,
    # 折叠展开
    "accordion": 200,
    # 分段控件指示器
    "indicator": 250,
    # 列表逐项入场
    "stagger_item": 300,
}

#: 逐项入场的间隔（毫秒）。规范要求 30-80ms 之间，过长会显得拖沓。
STAGGER_STEP_MS: Final[int] = 50

#: 进入动效的初始缩放值。
#:
#: 规范明令禁止从 ``scale(0)`` 开始——"现实中没有任何东西是从虚无中出现的"。
#: 不同元素的取值略有差异，越小越轻、越大越实。
ENTER_SCALE: Final[dict[str, float]] = {
    "popover": 0.95,
    "modal": 0.96,
    "tooltip": 0.97,
    "toast": 1.0,  # Toast 采用位移入场，不缩放
}

#: 按下时的缩放值。规范要求 0.95-0.98，保持细微。
PRESS_SCALE: Final[float] = 0.97

#: 触发"手滑关闭"的速度阈值（像素/毫秒）。
#: 有了速度判定，一次快速轻扫即可关闭，而不必拖过固定距离。
SWIPE_DISMISS_VELOCITY: Final[float] = 0.11

#: 通知默认停留时长（毫秒）
TOAST_DURATION_MS: Final[int] = 4000
#: 同时可见的通知条数上限
TOAST_VISIBLE_MAX: Final[int] = 3
#: 通知之间的间距（像素）
TOAST_GAP: Final[int] = 14
#: 通知距屏幕边缘的偏移（像素）
TOAST_OFFSET: Final[int] = 20

#: 首次工具提示的延迟（毫秒）。规范只规定"首次需延迟、后续立即"，
#: 未给出具体数值，此处取业界常用的 400ms：足以避免误触，又不至于让人等待。
TOOLTIP_INITIAL_DELAY_MS: Final[int] = 400


@dataclass
class MotionSpec:
    """动效参数集合，便于在"减少动态效果"模式下整体替换。"""

    enabled: bool = True
    duration: dict[str, int] = field(default_factory=lambda: dict(DURATION))
    easing: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: dict(EASING)
    )
    press_scale: float = PRESS_SCALE
    enter_scale: dict[str, float] = field(
        default_factory=lambda: dict(ENTER_SCALE)
    )
    stagger_step_ms: int = STAGGER_STEP_MS

    def scale_on_press(self) -> float:
        """按下缩放值；关闭动效或减少动态效果时返回 1.0（不缩放）。"""
        return self.press_scale if self.enabled else 1.0

    def duration_of(self, key: str) -> int:
        """取时长；关闭动效时返回 0（瞬时完成）。"""
        if not self.enabled:
            return 0
        return self.duration.get(key, self.duration["popover"])

    def easing_of(self, key: str) -> tuple[float, float, float, float]:
        """取缓动曲线控制点。"""
        return self.easing.get(key, self.easing["out"])


#: 全局动效参数。程序启动时依据系统"减少动态效果"设置改写。
#:
#: 规范要求："减少动态效果意味着**更少更柔和**，而不是完全取消——
#: 保留有助于理解的不透明度与颜色过渡，去掉位移与缩放。"
#: 因此该模式下 :attr:`MotionSpec.enabled` 仍为 ``True``，
#: 只是时长被压缩到接近瞬时、缩放被取消。
MOTION: Final[MotionSpec] = MotionSpec()

#: 减少动态效果模式下的时长上限（毫秒）。仅保留淡入淡出所需的最小时长。
REDUCED_MOTION_DURATION_MS: Final[int] = 120


def apply_reduced_motion(reduced: bool) -> None:
    """切换"减少动态效果"模式。

    依据规范，该模式下：

    - 位移与缩放动效被移除（``press_scale`` 归 1.0，进入缩放归 1.0）；
    - 不透明度与颜色过渡保留，但压缩到 :data:`REDUCED_MOTION_DURATION_MS`；
    - 逐项入场间隔归零（同时出现，不再有级联位移）。
    """
    MOTION.enabled = True
    MOTION.press_scale = 1.0 if reduced else PRESS_SCALE
    MOTION.stagger_step_ms = 0 if reduced else STAGGER_STEP_MS

    if reduced:
        for key in MOTION.duration:
            if MOTION.duration[key] > 0:
                MOTION.duration[key] = REDUCED_MOTION_DURATION_MS
        for key in MOTION.enter_scale:
            MOTION.enter_scale[key] = 1.0
    else:
        MOTION.duration.update(DURATION)
        MOTION.enter_scale.update(ENTER_SCALE)


# ==========================================================================
# 层级
# ==========================================================================

#: 控件层叠顺序。数值越大越靠上。
Z_ORDER: Final[dict[str, int]] = {
    "content": 0,
    "floating_bar": 10,
    "popover": 20,
    "toast": 30,
    "modal": 40,
    "overlay": 50,  # 原位翻译层始终位于最上层
}


# ==========================================================================
# 布局尺寸
# ==========================================================================

LAYOUT: Final[dict[str, int]] = {
    # 主面板
    "panel_width": 640,
    "panel_height": 560,
    # 设置行
    "row_height": 48,
    "row_min_height": 40,
    # 控件高度
    "control_height": 32,
    "button_height": 32,
    "input_height": 32,
    # 悬浮控制条
    "floating_bar_height": 36,
    "floating_bar_padding": 6,
    "floating_icon_size": 22,
    # 通知
    "toast_min_width": 280,
    "toast_max_width": 420,
    "toast_padding": 12,
    # 对话框
    "dialog_width": 520,
    "dialog_height": 420,
}


def contrast_ratio(foreground: str, background: str) -> float:
    """计算两色的 WCAG 对比度。

    用于在开发期校验令牌是否满足无障碍要求，也供单元测试断言使用。

    :return: 对比度，1.0 - 21.0。正文要求 ≥ 4.5，大字号要求 ≥ 3.0。
    """
    first = _relative_luminance(foreground)
    second = _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _relative_luminance(color: str) -> float:
    """计算 sRGB 颜色的相对亮度（WCAG 定义）。"""
    red, green, blue = _hex_to_rgb(color)

    def _channel(value: int) -> float:
        normalized = value / 255.0
        if normalized <= 0.03928:
            return normalized / 12.92
        return ((normalized + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * _channel(red) + 0.7152 * _channel(green) + 0.0722 * _channel(blue)
    )


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """把 ``#RRGGBB`` 转换为 ``(r, g, b)``。"""
    value = color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError(f"颜色格式无效：{color}")
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
    )
