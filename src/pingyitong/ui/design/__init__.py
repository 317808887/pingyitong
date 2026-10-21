"""设计系统：令牌、动效与主题。

设计来源与合规
--------------

**mozilla/OpenDesign**（MPL-2.0；设计作品 CC0-1.0）
    本项目采纳其"开放设计"原则：随程序交付可编辑的设计源文件
    （本目录的令牌文件与 ``docs/UI设计规范.md``），而非仅交付成品；
    界面如实呈现产品能力（"全离线、数据不出端"），不设诱导性文案；
    无障碍作为硬约束而非可选项（见 :func:`~pingyitong.ui.design.tokens.contrast_ratio`）。

**emilkowalski/skills**（设计工程规范）
    动效数值（时长、缓动曲线、缩放值、频率闸门）逐项取自该规范，
    并在 :mod:`~pingyitong.ui.design.motion` 中说明 Qt 与 CSS 的等价映射。

对外接口
--------

- :data:`~pingyitong.ui.design.tokens.MOTION` 全局动效参数
- :func:`~pingyitong.ui.design.tokens.apply_reduced_motion` 减少动态效果
- :func:`~pingyitong.ui.design.theme.build_theme` / :func:`~pingyitong.ui.design.theme.stylesheet`
- :func:`~pingyitong.ui.design.motion.enter` / :func:`~pingyitong.ui.design.motion.exit_widget`
"""

from __future__ import annotations

from .motion import (
    animate_geometry,
    animate_if_occasional,
    animate_opacity,
    animate_pos,
    animate_property,
    cubic_bezier,
    easing_for,
    enter,
    exit_widget,
    play_press_feedback,
    stagger_enter,
)
from .theme import Theme, base_font, build_theme, resolve_font_family, stylesheet
from .tokens import (
    BORDER_WIDTH,
    DARK_SCHEME,
    DURATION,
    EASING,
    ENTER_SCALE,
    FONT_FAMILIES,
    FONT_SIZE,
    FONT_WEIGHT,
    LAYOUT,
    LIGHT_SCHEME,
    MONO_FONT_FAMILIES,
    MOTION,
    PRESS_SCALE,
    RADIUS,
    SPACE,
    STAGGER_STEP_MS,
    SWIPE_DISMISS_VELOCITY,
    TOAST_DURATION_MS,
    TOAST_GAP,
    TOAST_OFFSET,
    TOAST_VISIBLE_MAX,
    TOOLTIP_INITIAL_DELAY_MS,
    Z_ORDER,
    ColorScheme,
    MotionSpec,
    apply_reduced_motion,
    contrast_ratio,
)

__all__ = [
    "BORDER_WIDTH",
    "ColorScheme",
    "DARK_SCHEME",
    "DURATION",
    "EASING",
    "ENTER_SCALE",
    "FONT_FAMILIES",
    "FONT_SIZE",
    "FONT_WEIGHT",
    "LAYOUT",
    "LIGHT_SCHEME",
    "MONO_FONT_FAMILIES",
    "MOTION",
    "MotionSpec",
    "PRESS_SCALE",
    "RADIUS",
    "SPACE",
    "STAGGER_STEP_MS",
    "SWIPE_DISMISS_VELOCITY",
    "TOAST_DURATION_MS",
    "TOAST_GAP",
    "TOAST_OFFSET",
    "TOAST_VISIBLE_MAX",
    "TOOLTIP_INITIAL_DELAY_MS",
    "Theme",
    "Z_ORDER",
    "animate_geometry",
    "animate_if_occasional",
    "animate_opacity",
    "animate_pos",
    "animate_property",
    "apply_reduced_motion",
    "base_font",
    "build_theme",
    "contrast_ratio",
    "cubic_bezier",
    "easing_for",
    "enter",
    "exit_widget",
    "play_press_feedback",
    "resolve_font_family",
    "stagger_enter",
    "stylesheet",
]
