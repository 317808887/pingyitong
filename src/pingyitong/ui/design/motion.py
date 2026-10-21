"""动效工具（Qt 实现）。

把设计规范中的动效规则落地为 Qt 可执行代码。规范原本面向 CSS/Web，
本模块负责等价映射，并保留其全部数值与约束。

映射关系
--------

======================  ==================================================
CSS 概念                 Qt 实现
======================  ==================================================
``cubic-bezier(a,b,c,d)``  :func:`cubic_bezier` → ``QEasingCurve``
``transition: transform``  ``QPropertyAnimation`` 作用于 ``geometry`` / ``pos``
``transform: scale(0.97)`` 通过 ``geometry`` 等比收缩实现（见 :func:`press_scale_animation`）
``opacity``               ``QGraphicsOpacityEffect`` 的 ``opacity`` 属性
``@keyframes``            ``QPropertyAnimation``（Qt 无关键帧重启问题，但本模块仍遵循
                          "动态界面用可中断的补间"这一原则——动画实例被复用而非重建）
``@media (prefers-reduced-motion)``  :func:`pingyitong.ui.design.tokens.apply_reduced_motion`
======================  ==================================================

规范中两条容易被忽略但必须遵守的约束
------------------------------------

1. **只动 ``transform`` 与 ``opacity``**。在 Qt 中对应"只动 ``geometry`` /
   ``pos`` 与不透明度"，绝不使用会触发重排的 ``setFixedWidth`` 逐帧动画。
   本模块因此只提供 :func:`animate_geometry`、:func:`animate_pos`、
   :func:`animate_opacity` 三类原语。

2. **高频操作不加动画**。规范把"键盘快捷键触发、每天 100 次以上"的操作
   列为硬性排除项（"No animation. Ever."）。本模块提供
   :func:`animate_if_occasional`，对这类动作直接瞬时完成，
   避免开发者无意中给热键动作加上过渡。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import (
    QEasingCurve,
    QObject,
    QPoint,
    QPropertyAnimation,
    QRect,
    QTimer,
)
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QWidget

from ...utils.logger import get_logger
from .tokens import MOTION, MotionSpec

logger = get_logger(__name__)


def _is_live(widget: QWidget) -> bool:
    """判断控件的底层 C++ 对象是否仍然存活。

    ``QTimer.singleShot`` 触发时控件可能早已随面板/对话框一起销毁，
    此时访问任何方法都会抛 ``RuntimeError: wrapped C/C++ object ... has
    been deleted``。PyQt6 对**定时器回调里的未捕获异常**会调用 ``qFatal()``
    直接终止进程（实测整个测试进程因此中断），所以凡是在延迟回调里
    碰控件的代码都必须先经过这里。
    """
    try:
        from PyQt6 import sip

        return not sip.isdeleted(widget)
    except RuntimeError:
        return False


def _bezier_curve_type() -> QEasingCurve.Type:
    """返回"自定义贝塞尔曲线"对应的枚举值。

    Qt 在 6.x 中把 ``QEasingCurve::Bezier`` 重命名为 ``BezierSpline``，
    因此这里按新旧两个名字依次探测，避免绑定版本差异导致运行时崩溃。
    实测 PyQt6 6.11 只提供 ``BezierSpline``——直接写 ``Bezier`` 会在
    第一次播放动效时抛 ``AttributeError``。
    """
    for name in ("BezierSpline", "Bezier"):
        member = getattr(QEasingCurve.Type, name, None)
        if member is not None:
            return member
    raise RuntimeError(
        "当前 PyQt6 版本未提供自定义贝塞尔曲线枚举（BezierSpline / Bezier）"
    )


def cubic_bezier(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> QEasingCurve:
    """构造与 CSS ``cubic-bezier(x1, y1, x2, y2)`` 等价的缓动曲线。

    CSS 的贝塞尔曲线从 ``(0, 0)`` 出发，经控制点 ``(x1, y1)``、``(x2, y2)``
    到达 ``(1, 1)``；Qt 的 ``addCubicBezierSegment`` 接受两个控制点与终点，
    起点隐含为 ``(0, 0)``，因此两者的参数可以直接对应，无需近似。
    """
    from PyQt6.QtCore import QPointF

    curve = QEasingCurve(_bezier_curve_type())
    curve.addCubicBezierSegment(
        QPointF(x1, y1),
        QPointF(x2, y2),
        QPointF(1.0, 1.0),
    )
    return curve


def easing_for(key: str, spec: MotionSpec | None = None) -> QEasingCurve:
    """按名称取缓动曲线。

    :param key: ``"out"`` / ``"in_out"`` / ``"drawer"``，见
        :data:`~pingyitong.ui.design.tokens.EASING`。
    """
    active = spec or MOTION
    x1, y1, x2, y2 = active.easing_of(key)
    return cubic_bezier(x1, y1, x2, y2)


# ----------------------------------------------------------------------
# 原语：几何 / 位置 / 不透明度
# ----------------------------------------------------------------------


def animate_geometry(
    target: QWidget,
    start: QRect,
    end: QRect,
    duration_ms: int,
    easing_key: str = "out",
    on_finished: Callable[[], None] | None = None,
    spec: MotionSpec | None = None,
) -> QPropertyAnimation | None:
    """对控件几何做补间动画。

    ``QWidget.geometry`` 同时涵盖位置与尺寸，是 Qt 中等价于 CSS
    ``transform`` 的最经济选择——它不会触发子控件重排，
    也不会像逐帧修改布局属性那样引起布局系统的反复计算。

    :return: 动画对象（已启动）；``duration_ms`` 为 0 时返回 ``None``。
    """
    if duration_ms <= 0:
        target.setGeometry(end)
        if on_finished is not None:
            on_finished()
        return None

    animation = QPropertyAnimation(target, b"geometry", target)
    animation.setStartValue(start)
    animation.setEndValue(end)
    animation.setDuration(duration_ms)
    animation.setEasingCurve(easing_for(easing_key, spec))
    if on_finished is not None:
        animation.finished.connect(on_finished)
    # 动画对象由 target 持有，随控件销毁而释放
    animation.start(QPropertyAnimation.DeletionPolicy.KeepWhenStopped)
    return animation


def animate_pos(
    target: QWidget,
    start: QPoint,
    end: QPoint,
    duration_ms: int,
    easing_key: str = "out",
    on_finished: Callable[[], None] | None = None,
    spec: MotionSpec | None = None,
) -> QPropertyAnimation | None:
    """对控件位置做补间动画（尺寸不变）。"""
    if duration_ms <= 0:
        target.move(end)
        if on_finished is not None:
            on_finished()
        return None

    animation = QPropertyAnimation(target, b"pos", target)
    animation.setStartValue(start)
    animation.setEndValue(end)
    animation.setDuration(duration_ms)
    animation.setEasingCurve(easing_for(easing_key, spec))
    if on_finished is not None:
        animation.finished.connect(on_finished)
    animation.start(QPropertyAnimation.DeletionPolicy.KeepWhenStopped)
    return animation


def ensure_opacity_effect(widget: QWidget) -> QGraphicsOpacityEffect:
    """确保控件具备不透明度效果并返回它。

    重复调用返回同一个效果对象，避免叠加多个效果导致渲染开销翻倍。
    """
    effect = widget.graphicsEffect()
    if isinstance(effect, QGraphicsOpacityEffect):
        return effect
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(1.0)
    widget.setGraphicsEffect(effect)
    return effect


def animate_opacity(
    target: QWidget,
    start: float,
    end: float,
    duration_ms: int,
    easing_key: str = "out",
    on_finished: Callable[[], None] | None = None,
    spec: MotionSpec | None = None,
) -> QPropertyAnimation | None:
    """对控件不透明度做补间动画。

    不透明度动画是"减少动态效果"模式下**唯一保留**的运动形式——
    规范要求减少动效时保留有助于理解的淡入淡出，只去掉位移与缩放。
    """
    effect = ensure_opacity_effect(target)

    if duration_ms <= 0:
        effect.setOpacity(end)
        if on_finished is not None:
            on_finished()
        return None

    animation = QPropertyAnimation(effect, b"opacity", target)
    animation.setStartValue(start)
    animation.setEndValue(end)
    animation.setDuration(duration_ms)
    animation.setEasingCurve(easing_for(easing_key, spec))
    if on_finished is not None:
        animation.finished.connect(on_finished)
    animation.start(QPropertyAnimation.DeletionPolicy.KeepWhenStopped)
    return animation


# ----------------------------------------------------------------------
# 组合：进入 / 退出
# ----------------------------------------------------------------------


@dataclass
class EnterExitSpec:
    """一次进入或退出动效的完整参数。"""

    duration_ms: int
    easing_key: str
    scale: float = 1.0
    fade: bool = True
    offset: QPoint | None = None


def enter(
    widget: QWidget,
    final_geometry: QRect,
    kind: str = "popover",
    on_finished: Callable[[], None] | None = None,
    spec: MotionSpec | None = None,
) -> None:
    """播放进入动效。

    初始状态为"缩小一点 + 透明"（绝不是 ``scale(0)``），
    目标是原尺寸与全不透明，符合"现实中不存在从虚无中出现的东西"。

    :param kind: ``popover`` / ``modal`` / ``tooltip`` / ``toast``，
        决定时长与初始缩放值。
    """
    active = spec or MOTION
    duration = active.duration_of(kind)
    scale = active.enter_scale.get(kind, 0.95)

    widget.setGeometry(_scaled_geometry(final_geometry, scale))
    if duration > 0:
        ensure_opacity_effect(widget).setOpacity(0.0)

    widget.show()
    widget.raise_()

    if duration <= 0:
        widget.setGeometry(final_geometry)
        ensure_opacity_effect(widget).setOpacity(1.0)
        if on_finished is not None:
            on_finished()
        return

    # 位置与不透明度同时补间。规范指出二者必须"合拍"，
    # 因此共用同一时长与缓动曲线。
    animate_geometry(
        widget,
        _scaled_geometry(final_geometry, scale),
        final_geometry,
        duration,
        "out",
        None,
        active,
    )
    animate_opacity(widget, 0.0, 1.0, duration, "out", on_finished, active)


def exit_widget(
    widget: QWidget,
    kind: str = "popover",
    hide_on_finish: bool = True,
    spec: MotionSpec | None = None,
) -> None:
    """播放退出动效并（可选）隐藏控件。

    规范指出退出应"沿进入的同一边缘、以同样速度"返回，
    因此这里复用与 :func:`enter` 相同的时长与缓动，只是方向相反。
    """
    active = spec or MOTION
    duration = active.duration_of(kind)

    if duration <= 0:
        if hide_on_finish:
            widget.hide()
        return

    scale = active.enter_scale.get(kind, 0.95)
    current = widget.geometry()

    animate_geometry(
        widget,
        current,
        _scaled_geometry(current, scale),
        duration,
        "out",
        (lambda: widget.hide()) if hide_on_finish else None,
        active,
    )
    animate_opacity(widget, 1.0, 0.0, duration, "out", None, active)


def _scaled_geometry(rect: QRect, scale: float) -> QRect:
    """以中心为锚点缩放矩形。

    规范要求弹出层"从触发器位置展开"，但 Qt 的顶层窗口无法可靠地设置
    ``transform-origin``；对于浮动在桌面上的窗口，以中心缩放是视觉上
    最不突兀的选择。模态对话框同样使用中心缩放（规范明确模态是例外）。
    """
    if scale >= 1.0:
        return rect

    width = max(1, int(rect.width() * scale))
    height = max(1, int(rect.height() * scale))
    center_x = rect.x() + rect.width() // 2
    center_y = rect.y() + rect.height() // 2
    return QRect(center_x - width // 2, center_y - height // 2, width, height)


# ----------------------------------------------------------------------
# 按钮按下反馈
# ----------------------------------------------------------------------


def animate_property(
    target: QObject,
    property_name: bytes,
    start: float,
    end: float,
    duration_ms: int,
    easing_key: str = "out",
    on_finished: Callable[[], None] | None = None,
    spec: MotionSpec | None = None,
) -> QPropertyAnimation | None:
    """对自定义 Qt 属性做补间动画。

    用于动画化控件内部的自绘状态（例如按钮的按下进度）。

    :param property_name: 属性名，必须是 ``pyqtProperty`` 声明的字节串，
        例如 ``b"pressProgress"``。
    """
    if duration_ms <= 0:
        target.setProperty(property_name.decode(), end)
        if on_finished is not None:
            on_finished()
        return None

    animation = QPropertyAnimation(target, property_name, target)
    animation.setStartValue(start)
    animation.setEndValue(end)
    animation.setDuration(duration_ms)
    animation.setEasingCurve(easing_for(easing_key, spec))
    if on_finished is not None:
        animation.finished.connect(on_finished)
    animation.start(QPropertyAnimation.DeletionPolicy.KeepWhenStopped)
    return animation


# ----------------------------------------------------------------------
# 按钮按下反馈
# ----------------------------------------------------------------------

#: 按下反馈的实现说明
#:
#: 规范要求所有可按压元素在按下时缩放到 0.97，时长 160ms、强 ease-out。
#: 这是让界面"看起来真的在听用户说话"的关键细节——几乎不会被察觉，
#: 但缺失时会明显感到迟滞。
#:
#: Qt 与 CSS 的差异：CSS 的 ``transform: scale()`` 由渲染引擎处理，
#: 不影响布局；Qt 若通过修改 ``geometry`` 模拟缩放，会被布局管理器
#: 立即纠正，因此**必须**在控件的 ``paintEvent`` 中通过
#: ``QPainter.scale()`` 实现。具体做法见
#: :class:`pingyitong.ui.widgets.button.Button`：它暴露一个
#: ``pressProgress`` 属性（0.0 - 1.0），由 :func:`animate_property`
#: 动画驱动，绘制时按 ``1 - 0.03 * pressProgress`` 缩放。
PRESS_SCALE_PROPERTY = b"pressProgress"


def play_press_feedback(
    widget: QObject,
    pressed: bool,
    spec: MotionSpec | None = None,
) -> QPropertyAnimation | None:
    """驱动控件的按下反馈动画。

    控件需实现 ``pressProgress`` 属性（0 = 未按下，1 = 完全按下）。
    减少动态效果模式下 ``scale_on_press()`` 返回 1.0，
    此时规范要求"保留反馈但不做位移与缩放"——因此仅播放极短的
    不透明度/颜色过渡（由控件自身在 paintEvent 中按 pressProgress
    调整背景色实现），不做几何变化。
    """
    active = spec or MOTION
    duration = active.duration_of("press")
    current = float(widget.property("pressProgress") or 0.0)
    target = 1.0 if pressed else 0.0

    if duration <= 0:
        widget.setProperty("pressProgress", target)
        return None

    return animate_property(
        widget,
        PRESS_SCALE_PROPERTY,
        current,
        target,
        duration,
        "out",
        None,
        active,
    )


# ----------------------------------------------------------------------
# 逐项入场
# ----------------------------------------------------------------------


def stagger_enter(
    widgets: list[QWidget],
    base_delay_ms: int = 0,
    spec: MotionSpec | None = None,
    on_item: Callable[[QWidget], None] | None = None,
) -> None:
    """让一组控件依次淡入，形成级联效果。

    规范指出：多个元素同时出现时应有 30-80ms 的间隔；同时强调
    "级联是装饰性的，绝不能在播放期间阻塞交互"——因此使用
    ``QTimer.singleShot`` 逐个触发，不占用事件循环，用户随时可以操作。
    """
    active = spec or MOTION
    step = active.stagger_step_ms
    duration = active.duration_of("stagger_item")

    for index, widget in enumerate(widgets):
        delay = base_delay_ms + index * step

        def _start(target: QWidget = widget) -> None:
            if not _is_live(target):
                return
            if duration <= 0:
                if on_item is not None:
                    on_item(target)
                return
            effect = ensure_opacity_effect(target)
            effect.setOpacity(0.0)
            if on_item is not None:
                on_item(target)
            animate_opacity(target, 0.0, 1.0, duration, "out", None, active)

        QTimer.singleShot(delay, _start)


def animate_if_occasional(
    widget: QWidget,
    final_geometry: QRect,
    kind: str = "popover",
    frequency: str = "occasional",
    spec: MotionSpec | None = None,
) -> None:
    """按操作频率决定是否播放动效。

    规范给出的频率闸门（不可协商）：

    ==========================================  ==========================
    频率                                        决策
    ==========================================  ==========================
    每天 100 次以上（快捷键、命令面板开关）      **永不加动画**
    每天数十次（悬停效果、列表导航）            移除或大幅削减
    偶尔（模态、抽屉、通知、设置）              标准动效
    罕见 / 首次（引导、空状态、成功反馈）        可以加入愉悦感
    ==========================================  ==========================

    :param frequency: ``"high"`` / ``"frequent"`` / ``"occasional"`` / ``"rare"``。
    """
    if frequency in {"high", "frequent"}:
        # 高频操作直接瞬时完成：动画会让人感觉"慢半拍"，且被反复观看
        widget.setGeometry(final_geometry)
        widget.show()
        widget.raise_()
        return

    enter(widget, final_geometry, kind, None, spec)
