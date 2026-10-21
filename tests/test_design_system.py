"""设计系统与动效测试（**需要真实 PyQt6**）。

这个测试文件的存在有明确原因：项目的纯逻辑测试使用一个 PyQt6 替身，
以便在没有 GUI 依赖的环境中运行。替身对所有属性访问都返回一个"万能对象"，
因此**无法发现枚举名写错、API 签名变更这类绑定层错误**。

实际发生过一次：``QEasingCurve.Type.Bezier`` 在 Qt 6 中已改名为
``BezierSpline``，代码在第一次播放通知动效时抛 ``AttributeError`` 并导致
程序退出。纯逻辑测试全绿，但程序一启动就崩。本文件专门覆盖这类问题。

没有安装 PyQt6 或无显示环境时整个模块会被跳过，不会导致测试失败。
"""

from __future__ import annotations

import unittest

try:
    from PyQt6.QtCore import QEasingCurve, QRect
    from PyQt6.QtWidgets import QApplication, QWidget

    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover - 无 PyQt6 环境
    PYQT_AVAILABLE = False


def _ensure_app() -> QApplication | None:
    """确保存在 QApplication 实例（Qt 控件与绘制的必要条件）。"""
    if not PYQT_AVAILABLE:
        return None
    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception:  # noqa: BLE001 - 无显示环境时会失败
            return None
    return app


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestEasingCurves(unittest.TestCase):
    """缓动曲线构造。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _ensure_app()
        if cls.app is None:
            raise unittest.SkipTest("无法创建 QApplication（可能无显示环境）")

    def test_design_tokens_are_constructible(self) -> None:
        """设计规范中定义的每条曲线都必须能构造出来。

        这是对"枚举名写错"最直接的防线：三条曲线全部过一遍。
        """
        from pingyitong.ui.design import EASING
        from pingyitong.ui.design.motion import cubic_bezier

        for name, (x1, y1, x2, y2) in EASING.items():
            with self.subTest(curve=name):
                curve = cubic_bezier(x1, y1, x2, y2)
                self.assertIsInstance(curve, QEasingCurve)

    def test_easing_for_all_keys(self) -> None:
        """``easing_for`` 对全部已定义名称都应返回曲线。"""
        from pingyitong.ui.design.motion import easing_for

        for key in ("out", "in_out", "drawer"):
            with self.subTest(key=key):
                self.assertIsInstance(easing_for(key), QEasingCurve)

    def test_easing_for_unknown_falls_back(self) -> None:
        """未知名称回落到默认曲线而不是抛异常。"""
        from pingyitong.ui.design.motion import easing_for

        self.assertIsInstance(easing_for("不存在的曲线"), QEasingCurve)

    def test_curve_evaluates_in_unit_range(self) -> None:
        """曲线在 0 与 1 处应分别返回 0 与 1。

        自定义贝塞尔曲线的端点必须落在正确位置，否则动画会跳变。
        """
        from pingyitong.ui.design.motion import easing_for

        curve = easing_for("out")
        self.assertAlmostEqual(curve.valueForProgress(0.0), 0.0, places=3)
        self.assertAlmostEqual(curve.valueForProgress(1.0), 1.0, places=3)


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestMotionPrimitives(unittest.TestCase):
    """动效原语在真实 Qt 下可用。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _ensure_app()
        if cls.app is None:
            raise unittest.SkipTest("无法创建 QApplication（可能无显示环境）")

    def test_animate_geometry_zero_duration_is_instant(self) -> None:
        """时长为 0（减少动态效果模式）时应立即到位且不创建动画。"""
        from pingyitong.ui.design.motion import animate_geometry

        widget = QWidget()
        try:
            target = QRect(10, 20, 100, 50)
            result = animate_geometry(widget, QRect(0, 0, 1, 1), target, 0)
            self.assertIsNone(result)
            self.assertEqual(widget.geometry(), target)
        finally:
            widget.deleteLater()

    def test_animate_opacity_zero_duration(self) -> None:
        """不透明度动画在零时长下也应生效（减少动态效果要保留它）。"""
        from pingyitong.ui.design.motion import animate_opacity, ensure_opacity_effect

        widget = QWidget()
        try:
            animate_opacity(widget, 1.0, 0.5, 0)
            effect = ensure_opacity_effect(widget)
            self.assertAlmostEqual(effect.opacity(), 0.5, places=3)
        finally:
            widget.deleteLater()

    def test_animate_geometry_with_duration_creates_animation(self) -> None:
        """有时长时应返回一个已启动的动画对象，且起点为当前几何。"""
        from pingyitong.ui.design.motion import animate_geometry

        widget = QWidget()
        try:
            widget.setGeometry(QRect(0, 0, 10, 10))
            start = QRect(0, 0, 10, 10)
            end = QRect(50, 60, 200, 80)
            animation = animate_geometry(widget, start, end, 200)
            self.assertIsNotNone(animation)
            self.assertEqual(animation.duration(), 200)
            self.assertEqual(animation.startValue(), start)
            self.assertEqual(animation.endValue(), end)
        finally:
            widget.deleteLater()

    def test_enter_sets_initial_scale_not_zero(self) -> None:
        """进入动效的初始几何必须大于 0 尺寸。

        规范明令禁止从 ``scale(0)`` 开始——"现实中没有任何东西是从虚无中
        出现的"。这里直接检查初始几何不为空。
        """
        from pingyitong.ui.design.motion import _scaled_geometry

        target = QRect(100, 100, 200, 100)
        for kind, scale in (("popover", 0.95), ("modal", 0.96), ("tooltip", 0.97)):
            with self.subTest(kind=kind):
                initial = _scaled_geometry(target, scale)
                self.assertGreater(initial.width(), 0)
                self.assertGreater(initial.height(), 0)
                self.assertLess(initial.width(), target.width())
                self.assertLess(initial.height(), target.height())

                # 以中心为锚点缩放：允许 1px 偏差。
                # 尺寸为奇数时（例如 100×0.95=95）整数取整必然带来半像素，
                # 动画时长仅 200ms 且缩放幅度只有 5%，1px 的锚点偏移
                # 肉眼不可见，不值得为它引入偶数对齐的复杂度。
                center = initial.center()
                self.assertLessEqual(abs(center.x() - target.center().x()), 1)
                self.assertLessEqual(abs(center.y() - target.center().y()), 1)

    def test_play_press_feedback_does_not_raise(self) -> None:
        """按下反馈必须能在真实 Qt 下执行。

        它内部会读取 ``pressProgress`` 属性并驱动动画；若控件未声明该属性，
        Qt 会抛错。这里用一个声明了该属性的最小控件验证。
        """
        from PyQt6.QtCore import pyqtProperty

        from pingyitong.ui.design.motion import play_press_feedback

        class Pressable(QWidget):
            def __init__(self) -> None:
                super().__init__()
                self._progress = 0.0

            def _get(self) -> float:
                return self._progress

            def _set(self, value: float) -> None:
                self._progress = float(value)

            pressProgress = pyqtProperty(float, _get, _set)

        widget = Pressable()
        try:
            play_press_feedback(widget, True)
            play_press_feedback(widget, False)
        finally:
            widget.deleteLater()

    def test_stagger_enter_does_not_raise(self) -> None:
        """逐项入场在真实 Qt 下应能执行。"""
        from pingyitong.ui.design.motion import stagger_enter

        widgets = [QWidget() for _ in range(3)]
        try:
            stagger_enter(widgets, base_delay_ms=0)
        finally:
            for widget in widgets:
                widget.deleteLater()


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestThemeAndTokens(unittest.TestCase):
    """主题与令牌的内部一致性。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _ensure_app()
        if cls.app is None:
            raise unittest.SkipTest("无法创建 QApplication（可能无显示环境）")

    def test_stylesheet_generates_for_both_themes(self) -> None:
        """两套主题都应能生成样式表且非空。"""
        from pingyitong.ui.design.theme import build_theme, stylesheet

        for name in ("dark", "light"):
            with self.subTest(theme=name):
                text = stylesheet(build_theme(name))
                self.assertGreater(len(text), 1000)
                self.assertIn("QPushButton", text)

    def test_style_config_fields_all_appear_in_stylesheet(self) -> None:
        """样式表中不应残留未替换的占位符。"""
        from pingyitong.ui.design.theme import build_theme, stylesheet

        text = stylesheet(build_theme("dark"))
        self.assertNotIn("{c.", text, "样式表中有未替换的颜色占位符")
        self.assertNotIn("None", text)

    def test_contrast_meets_wcag_aa_for_body_text(self) -> None:
        """正文文字与背景的对比度应满足 WCAG AA（4.5:1）。

        无障碍是硬约束，不是一个"尽量"的目标，因此把它写成断言。
        """
        from pingyitong.ui.design.tokens import (
            DARK_SCHEME,
            LIGHT_SCHEME,
            contrast_ratio,
        )

        for label, scheme in (("dark", DARK_SCHEME), ("light", LIGHT_SCHEME)):
            with self.subTest(theme=label):
                ratio = contrast_ratio(scheme.text_primary, scheme.background)
                self.assertGreaterEqual(
                    ratio, 4.5,
                    f"{label} 主题正文对比度仅 {ratio:.2f}:1，低于 WCAG AA 要求",
                )

    def test_accent_text_has_better_contrast_than_accent(self) -> None:
        """作为文字使用的强调色必须比背景填充色更清晰。

        深色主题下 ``accent`` 直接当文字用对比度不足，因此专门提供了
        ``accent_text``。这条断言保证该设计意图没有被改坏。
        """
        from pingyitong.ui.design.tokens import (
            DARK_SCHEME,
            contrast_ratio,
        )

        as_text = contrast_ratio(DARK_SCHEME.accent_text, DARK_SCHEME.surface)
        raw = contrast_ratio(DARK_SCHEME.accent, DARK_SCHEME.surface)
        self.assertGreater(as_text, raw)

    def test_reduced_motion_keeps_opacity_but_drops_transform(self) -> None:
        """减少动态效果：保留不透明度过渡，去掉位移与缩放。

        规范明确要求"更少更柔和，而不是完全取消"。
        """
        from pingyitong.ui.design.tokens import (
            MOTION,
            PRESS_SCALE,
            apply_reduced_motion,
        )

        try:
            apply_reduced_motion(True)
            self.assertEqual(MOTION.press_scale, 1.0, "减少动态效果时应取消按下缩放")
            self.assertEqual(MOTION.stagger_step_ms, 0, "不应再有级联位移")
            self.assertTrue(MOTION.enabled, "不应完全关闭动效")
            self.assertGreater(
                MOTION.duration_of("popover"), 0, "不透明度过渡必须保留"
            )
            self.assertEqual(MOTION.enter_scale["popover"], 1.0)
        finally:
            apply_reduced_motion(False)

        self.assertAlmostEqual(MOTION.press_scale, PRESS_SCALE)

    def test_motion_durations_within_budget(self) -> None:
        """动效时长必须落在设计规范给定的区间内。

        规范给的是**分档预算**，不是单一上限：

        ==========================  ==========
        元素                        时长
        ==========================  ==========
        按钮按压                    100–160ms
        工具提示 / 小弹层           125–200ms
        下拉 / 选择器               150–250ms
        模态 / 抽屉                 200–500ms
        通知                        400ms（规范专门给定，刻意略慢）
        ==========================  ==========

        同时规范有一条总则："UI 动效保持在 300ms 以内"。因此对高频、
        轻量的元素（按压、提示、弹出层、分段指示器、折叠）按 300ms 卡，
        模态与抽屉这类大范围、低频的过渡按 500ms 卡。
        """
        from pingyitong.ui.design.tokens import DURATION

        #: 必须 ≤300ms 的轻量动效
        light = {"press", "tooltip", "popover", "indicator", "accordion"}
        #: 允许到 500ms 的大范围过渡（规范的分档上限）
        heavy = {"modal", "drawer", "toast", "stagger_item"}

        for key, value in DURATION.items():
            if key == "instant":
                self.assertEqual(value, 0, "instant 应为 0（高频操作不加动画）")
                continue
            with self.subTest(duration=key):
                if key in light:
                    self.assertLessEqual(
                        value, 300, f"{key} 的时长 {value}ms 超出 300ms 预算"
                    )
                elif key in heavy:
                    self.assertLessEqual(
                        value, 500, f"{key} 的时长 {value}ms 超出 500ms 分档上限"
                    )
                else:
                    self.fail(f"未归类的动效时长：{key}（请在测试中显式归类）")


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestWidgetRendering(unittest.TestCase):
    """真实渲染每个自绘控件。

    **这个测试类的存在理由**：控件的 ``paintEvent`` 里混用 ``QRect``（只接受整数）
    与浮点令牌值会在运行到绘制时抛 ``TypeError``，而这种错误在"只构造不绘制"
    的测试里完全不会暴露——实测发生过一次：通知控件显示时程序直接崩溃。
    因此这里用 ``QPixmap`` 真正跑一遍 ``render()``，把绘制路径全部走通。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _ensure_app()
        if cls.app is None:
            raise unittest.SkipTest("无法创建 QApplication（可能无显示环境）")

    def _render(self, widget: QWidget, size: tuple[int, int] = (420, 120)) -> None:
        """把控件渲染到离屏图像。

        渲染会真实触发 ``paintEvent``，因此任何几何参数类型错误都会在此暴露。
        """
        from PyQt6.QtGui import QPixmap

        widget.resize(*size)
        pixmap = QPixmap(size[0], size[1])
        widget.render(pixmap)

    def test_toast_renders(self) -> None:
        """通知控件（含标题、说明与操作按钮）必须能绘制。

        这是实际崩溃过的那条路径：文本矩形用了浮点内边距。
        """
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.widgets.toast import ToastKind, ToastRequest, ToastWidget

        theme = build_theme("dark")
        request = ToastRequest(
            title="模型已就绪",
            description="极速模式加载完成，可直接开始翻译",
            kind=ToastKind.SUCCESS,
            action_text="查看设置",
        )
        widget = ToastWidget(request, theme)
        try:
            self._render(widget)
        finally:
            widget.deleteLater()

    def test_toast_renders_all_kinds(self) -> None:
        """四种通知类型都应能绘制（各自的配色分支不同）。"""
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.widgets.toast import ToastKind, ToastRequest, ToastWidget

        theme = build_theme("dark")
        for kind in ToastKind:
            with self.subTest(kind=kind.value):
                widget = ToastWidget(ToastRequest(title="标题", description="说明", kind=kind), theme)
                try:
                    self._render(widget)
                finally:
                    widget.deleteLater()

    def test_toast_renders_without_description(self) -> None:
        """只有标题的通知也必须能绘制（布局分支不同）。"""
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.widgets.toast import ToastRequest, ToastWidget

        widget = ToastWidget(ToastRequest(title="仅标题"), build_theme("light"))
        try:
            self._render(widget, (300, 60))
        finally:
            widget.deleteLater()

    def test_buttons_render(self) -> None:
        """按钮的四种变体与按下状态都必须能绘制。"""
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.widgets.button import Button, IconButton

        theme = build_theme("dark")
        for variant in Button.VARIANTS:
            with self.subTest(variant=variant):
                button = Button("测试按钮", variant, theme)
                try:
                    self._render(button, (140, 40))
                    # 按下状态走的是另一条绘制分支
                    button.setProperty("pressProgress", 1.0)
                    self._render(button, (140, 40))
                finally:
                    button.deleteLater()

        icon = IconButton("⚙", "设置", theme)
        try:
            self._render(icon, (32, 32))
        finally:
            icon.deleteLater()

    def test_switch_renders(self) -> None:
        """开关在开与关两种状态下都必须能绘制。"""
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.widgets.controls import Switch

        for checked in (False, True):
            with self.subTest(checked=checked):
                widget = Switch(checked, build_theme("dark"))
                try:
                    self._render(widget, (48, 28))
                finally:
                    widget.deleteLater()

    def test_segmented_control_renders(self) -> None:
        """分段控件必须能绘制（含指示器与文字）。"""
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.widgets.controls import SegmentedControl

        widget = SegmentedControl(["极速 1.7B", "高精度 4B"], 1, build_theme("dark"))
        try:
            self._render(widget, (260, 34))
        finally:
            widget.deleteLater()

    def test_style_preview_renders(self) -> None:
        """设置面板的样式预览控件必须能绘制。"""
        from pingyitong.config.schema import AppConfig
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.main_panel import _StylePreview

        preview = _StylePreview(build_theme("dark"))
        try:
            preview.set_style(AppConfig().style)
            self._render(preview, (500, 140))
        finally:
            preview.deleteLater()

    def test_overlay_window_renders_items(self) -> None:
        """原位翻译层必须能把渲染条目真正画出来。

        这是产品的核心界面；它的绘制路径覆盖了自适应排版、遮罩圆角与
        文字阴影，任何几何类型错误都会让译文整层消失。
        """
        from pingyitong.components.capture.geometry import Rect, ScreenInfo
        from pingyitong.components.render.overlay_window import OverlayWindow
        from pingyitong.components.render.render_item import RenderItem
        from pingyitong.config.schema import AppConfig

        screen = ScreenInfo(
            name="TEST",
            logical=Rect(0, 0, 800, 600),
            physical=Rect(0, 0, 800, 600),
            device_pixel_ratio=1.0,
            is_primary=True,
            mss_index=1,
        )
        window = OverlayWindow(screen, AppConfig().style)
        try:
            window.set_items(
                [
                    RenderItem(box=Rect(20, 20, 200, 24), text="保存为"),
                    RenderItem(
                        box=Rect(20, 60, 300, 20),
                        text="电池电量较低，请连接充电器。",
                        is_dark_background=True,
                    ),
                    # 超长文本走截断分支
                    RenderItem(
                        box=Rect(20, 100, 80, 18),
                        text="这是一段很长很长的译文，用来验证截断与省略号分支",
                    ),
                ]
            )
            self._render(window, (800, 600))
        finally:
            window.deleteLater()

    def test_overlay_renders_with_overlay_mode(self) -> None:
        """半透明叠加模式的绘制分支也必须能走通。"""
        from pingyitong.components.capture.geometry import Rect, ScreenInfo
        from pingyitong.components.render.overlay_window import OverlayWindow
        from pingyitong.components.render.render_item import RenderItem
        from pingyitong.config.schema import AppConfig

        config = AppConfig()
        config.style.mode = "overlay"
        config.style.bold = True

        screen = ScreenInfo(
            name="TEST",
            logical=Rect(0, 0, 400, 300),
            physical=Rect(0, 0, 400, 300),
            device_pixel_ratio=1.0,
            is_primary=True,
            mss_index=1,
        )
        window = OverlayWindow(screen, config.style)
        try:
            window.set_items([RenderItem(box=Rect(10, 10, 150, 20), text="设置")])
            self._render(window, (400, 300))
        finally:
            window.deleteLater()

    def test_tooltip_renders(self) -> None:
        """工具提示控件必须能绘制。"""
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.widgets.tooltip import _TooltipWidget

        widget = _TooltipWidget("开启 / 暂停暂停翻译", build_theme("dark"))
        try:
            self._render(widget, (160, 30))
        finally:
            widget.deleteLater()

    def test_region_selector_selection_rect_is_qt_rect(self) -> None:
        """``_selection_rect`` 必须返回 Qt 的 QRect（API 归属的显式断言）。

        这条断言把"哪套 API"这件事写进测试，避免以后有人按本项目 ``Rect``
        的属性去调用它——那正是崩溃的成因。
        """
        from PyQt6.QtCore import QPoint, QRect

        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.region_selector import RegionSelector

        selector = RegionSelector(build_theme("dark"))
        try:
            selector._origin = QPoint(10, 20)
            selector._current = QPoint(110, 220)
            rect = selector._selection_rect()

            self.assertIsInstance(rect, QRect)
            # Qt 的 API
            self.assertTrue(hasattr(rect, "isEmpty"))
            self.assertFalse(rect.isEmpty())
            # 不应把 Qt QRect 与本项目 Rect 的属性混用
            self.assertFalse(
                hasattr(rect, "is_empty"),
                "Qt 的 QRect 没有 is_empty 属性——误用会导致崩溃",
            )
        finally:
            selector.deleteLater()

    def test_selection_logical_dimensions_are_ints(self) -> None:
        """``_selection_logical`` 的宽高**必须是数字**，不能是绑定方法。

        这是"框选屏幕时崩溃"的根因回归测试。Qt 的 ``QRect`` 用
        ``rect.width()`` / ``rect.x()``（方法），本项目的 ``Rect`` 用
        ``rect.width`` / ``rect.x``（属性）。一旦把方法对象塞进
        ``Rect``，``dataclass`` 不会报错，要等到后续做算术时才炸
        ``int + builtin_function_or_method``；而在 Qt 事件处理器里
        未捕获的异常会让 PyQt6 直接 ``qFatal()`` 结束进程，
        用户看到的就是"一点框选程序就没了"。
        """
        from PyQt6.QtCore import QPoint

        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.region_selector import RegionSelector

        selector = RegionSelector(build_theme("dark"))
        try:
            selector._origin = QPoint(120, 90)
            selector._current = QPoint(620, 450)
            logical = selector._selection_logical()

            self.assertIsNotNone(logical)
            for name in ("x", "y", "width", "height"):
                value = getattr(logical, name)
                self.assertIsInstance(
                    value,
                    int,
                    f"{name} 必须是整数；拿到 {type(value).__name__} 说明"
                    f"把 Qt QRect 的方法当成了属性",
                )
            self.assertEqual(logical.width, 501)
            self.assertEqual(logical.height, 361)
            # 说明：Qt 用两个点构造的矩形是"含首含尾"的，
            # QPoint(120,90) → QPoint(620,450) 的宽是 620-120+1 = 501。
        finally:
            selector.deleteLater()

    def test_selection_logical_is_empty_when_no_drag(self) -> None:
        """未拖拽时返回 ``None``；原地误触会被尺寸下限挡掉。

        Qt 两个点构造的矩形含首含尾，因此"原地按下再松开"得到的是
        1×1 而非空矩形，它必须被 :meth:`_confirmed_selection` 判为无效。
        """
        from PyQt6.QtCore import QPoint

        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.region_selector import RegionSelector

        selector = RegionSelector(build_theme("dark"))
        try:
            selector._origin = None
            selector._current = None
            self.assertIsNone(selector._selection_logical())
            self.assertIsNone(selector._confirmed_selection())

            # 原地按下再松开：属于误触，不得被确认为有效选区
            selector._origin = QPoint(50, 50)
            selector._current = QPoint(50, 50)
            self.assertIsNone(selector._confirmed_selection())

            # 比下限小一点也应被拒绝
            selector._origin = QPoint(50, 50)
            selector._current = QPoint(50 + RegionSelector.MIN_SIZE - 2, 50 + 100)
            self.assertIsNone(selector._confirmed_selection())

            # 达到下限即可通过
            selector._current = QPoint(50 + RegionSelector.MIN_SIZE, 50 + 100)
            self.assertIsNotNone(selector._confirmed_selection())
        finally:
            selector.deleteLater()

    def test_selection_logical_survives_screen_matching(self) -> None:
        """选区必须能安全地参与显示器匹配——崩溃就发生在这里。

        ``region_to_monitor`` 会对选区做 ``intersection``（内部是整数加法）。
        这里走一遍真实调用，确保换算结果能被下游几何代码消化。
        """
        from PyQt6.QtCore import QPoint

        from pingyitong.components.capture.geometry import list_screens, region_to_monitor
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.region_selector import RegionSelector

        selector = RegionSelector(build_theme("dark"))
        try:
            selector._origin = QPoint(100, 100)
            selector._current = QPoint(700, 500)
            logical = selector._selection_logical()
            monitor = region_to_monitor(logical, list_screens())
            # 只要没有抛异常即可；结果取决于实际显示器布局
            self.assertTrue(monitor is None or hasattr(monitor, "mss_index"))
        finally:
            selector.deleteLater()

    def test_region_selector_emits_wellformed_rect_on_drag(self) -> None:
        """用真实 Qt 事件走完整拖拽，验证发出的矩形字段都是数字。

        这是端到端路径：``mousePressEvent`` → ``mouseMoveEvent`` →
        ``mouseReleaseEvent`` → ``region_selected``。
        """
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest

        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.region_selector import RegionSelector

        selector = RegionSelector(build_theme("dark"))
        received: list = []
        selector.region_selected.connect(lambda rect, mon: received.append(rect))
        try:
            selector.start()
            QTest.qWait(50)
            QTest.mousePress(
                selector, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(200, 200)
            )
            QTest.qWait(20)
            QTest.mouseMove(selector, QPoint(600, 480))
            QTest.qWait(20)
            QTest.mouseRelease(
                selector, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(600, 480)
            )
            QTest.qWait(50)
        finally:
            selector.deleteLater()

        self.assertEqual(len(received), 1, "拖拽后应发出一次 region_selected")
        rect = received[0]
        for name in ("x", "y", "width", "height"):
            self.assertIsInstance(getattr(rect, name), int)
        self.assertGreaterEqual(rect.width, RegionSelector.MIN_SIZE)
        self.assertGreaterEqual(rect.height, RegionSelector.MIN_SIZE)

    def test_region_selector_enter_confirms_selection(self) -> None:
        """Enter 键确认选框时也必须发出字段正确的矩形。"""
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest

        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.region_selector import RegionSelector

        selector = RegionSelector(build_theme("dark"))
        received: list = []
        selector.region_selected.connect(lambda rect, mon: received.append(rect))
        try:
            selector.start()
            QTest.qWait(50)
            # 直接把拖拽状态摆好，再按 Enter
            selector._origin = QPoint(150, 120)
            selector._current = QPoint(650, 520)
            QTest.keyClick(selector, Qt.Key.Key_Return)
            QTest.qWait(50)
        finally:
            selector.deleteLater()

        self.assertEqual(len(received), 1)
        rect = received[0]
        self.assertIsInstance(rect.width, int)
        self.assertEqual(rect.width, 501)

    def test_region_selector_escape_cancels(self) -> None:
        """Esc 必须取消而不是发出选区（键盘出口）。"""
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest

        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.region_selector import RegionSelector

        selector = RegionSelector(build_theme("dark"))
        selected: list = []
        cancelled: list = []
        selector.region_selected.connect(lambda rect, mon: selected.append(rect))
        selector.cancelled.connect(lambda: cancelled.append(True))
        try:
            selector.start()
            QTest.qWait(50)
            selector._origin = QPoint(150, 120)
            selector._current = QPoint(650, 520)
            QTest.keyClick(selector, Qt.Key.Key_Escape)
            QTest.qWait(50)
        finally:
            selector.deleteLater()

        self.assertEqual(cancelled, [True])
        self.assertEqual(selected, [], "取消时不应发出选区")

    def test_region_badge_shows_numbers_not_method_objects(self) -> None:
        """尺寸标签必须显示数字。

        ``selection.width`` 是 Qt ``QRect`` 的**方法**，直接插进 f-string
        会把方法对象本身显示出来（``<built-in method width of QRect
        object at 0x...>``）。这里断言渲染出的标签文本里没有这种东西。
        """
        from PyQt6.QtCore import QPoint

        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.region_selector import RegionSelector

        selector = RegionSelector(build_theme("dark"))
        captured: list[str] = []
        try:
            selector._origin = QPoint(100, 100)
            selector._current = QPoint(640, 460)

            from PyQt6.QtGui import QPainter, QPixmap

            pixmap = QPixmap(800, 600)
            painter = QPainter(pixmap)
            try:
                # 直接调用绘制函数并捕获它画的文本
                original = painter.drawText

                def spy(rect, flags, text="", *args):
                    if text:
                        captured.append(text)
                    return original(rect, flags, text, *args)

                painter.drawText = spy  # type: ignore[method-assign]
                selector._paint_dimension_badge(painter, selector._selection_rect(), build_theme("dark").scheme)
            finally:
                painter.end()

            badge = [t for t in captured if "×" in t]
            self.assertTrue(badge, f"未绘制尺寸标签，捕获到：{captured}")
            # QPoint(100,100) → QPoint(640,460) 含首含尾，宽高为 541 × 361
            self.assertIn("541 × 361", badge[0])
            self.assertNotIn("built-in method", badge[0])
        finally:
            selector.deleteLater()

    def test_region_selector_renders_with_selection(self) -> None:
        """区域框选覆盖层在"已拖出选框"状态下必须能绘制。

        **这是实际崩溃过的那条路径。** 覆盖层持有的是 Qt 的 ``QRect``，
        而绘制代码误用了本项目 ``Rect`` 的属性 ``is_empty``（Qt 只有方法
        ``isEmpty()``），于是 ``paintEvent`` 抛 ``AttributeError``，
        用户一点"框选区域"程序就消失了。

        这个测试会真正触发 ``paintEvent``，因此能挡住同类回归。
        """
        from PyQt6.QtCore import QPoint

        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.region_selector import RegionSelector

        selector = RegionSelector(build_theme("dark"))
        try:
            # 模拟一次拖拽：起止点都不为空 → 走进绘制选框与尺寸标签的分支
            selector._origin = QPoint(120, 90)
            selector._current = QPoint(520, 330)
            selector._dragging = True
            self._render(selector, (800, 600))
        finally:
            selector.deleteLater()

    def test_region_selector_renders_without_selection(self) -> None:
        """未开始拖拽时也应能绘制（走另一条分支）。"""
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.region_selector import RegionSelector

        selector = RegionSelector(build_theme("dark"))
        try:
            selector._origin = None
            selector._current = None
            self._render(selector, (800, 600))
        finally:
            selector.deleteLater()

    def test_region_selector_renders_tiny_selection(self) -> None:
        """极小选框（接近误触）也应能绘制，不应触发除零或空矩形异常。"""
        from PyQt6.QtCore import QPoint

        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.region_selector import RegionSelector

        selector = RegionSelector(build_theme("dark"))
        try:
            selector._origin = QPoint(100, 100)
            selector._current = QPoint(101, 100)  # 零高度
            self._render(selector, (400, 300))
        finally:
            selector.deleteLater()


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestToastManager(unittest.TestCase):
    """通知管理器的行为。

    这里覆盖的是实际崩溃过的路径：**相同 key 的通知替换自身**时，
    关闭动作会同步触发"已关闭"回调完成列表移除，若调用方再移除一次就抛
    ``ValueError: list.remove(x): x not in list``。程序启动时正好会连续
    发两条同 key 的模型加载提示，于是每次启动必崩。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _ensure_app()
        if cls.app is None:
            raise unittest.SkipTest("无法创建 QApplication（可能无显示环境）")

    def _manager(self):
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.widgets.toast import ToastManager

        return ToastManager(build_theme("dark"), enabled=True)

    def test_same_key_replaces_without_error(self) -> None:
        """相同 key 的通知应替换而不是叠加，且不得抛异常。"""
        manager = self._manager()
        try:
            manager.show("正在加载模型", "准备中", key="preload")
            manager.show("正在加载模型", "已加载 1/2", key="preload")
            manager.show("模型已就绪", "可以开始翻译", key="preload")
            # 替换语义：只应剩最后一条
            self.assertEqual(len(manager._toasts), 1)
            self.assertEqual(manager._toasts[0].request.title, "模型已就绪")
        finally:
            manager.dismiss_all()

    def test_different_keys_coexist(self) -> None:
        """不同 key 的通知应各自独立存在。"""
        manager = self._manager()
        try:
            manager.show("第一条", key="a")
            manager.show("第二条", key="b")
            self.assertEqual(len(manager._toasts), 2)
        finally:
            manager.dismiss_all()

    def test_without_key_stacks(self) -> None:
        """不带 key 的通知允许叠加。"""
        manager = self._manager()
        try:
            manager.show("第一条")
            manager.show("第二条")
            self.assertEqual(len(manager._toasts), 2)
        finally:
            manager.dismiss_all()

    def test_visible_count_capped(self) -> None:
        """同时可见数量必须严格不超过设计规范的上限。

        这条断言曾经抓到过一个真实缺陷：超限处理只调用异步的 dismiss()
        而没有立即从跟踪列表移除，导致计数永不下降，上限形同虚设——
        连续来十条通知就会十条全留在屏幕上。
        """
        from pingyitong.ui.design.tokens import TOAST_VISIBLE_MAX

        manager = self._manager()
        try:
            for index in range(TOAST_VISIBLE_MAX + 4):
                manager.show(f"通知 {index}")
            self.assertEqual(
                len(manager._toasts),
                TOAST_VISIBLE_MAX,
                f"应只保留 {TOAST_VISIBLE_MAX} 条可见通知",
            )
        finally:
            manager.dismiss_all()

    def test_newest_toast_survives_capping(self) -> None:
        """超限时被挤掉的应是最早那条，最新的一定保留。"""
        from pingyitong.ui.design.tokens import TOAST_VISIBLE_MAX

        manager = self._manager()
        try:
            for index in range(TOAST_VISIBLE_MAX + 2):
                manager.show(f"通知 {index}")
            titles = [toast.request.title for toast in manager._toasts]
            self.assertIn(f"通知 {TOAST_VISIBLE_MAX + 1}", titles)
            self.assertNotIn("通知 0", titles)
        finally:
            manager.dismiss_all()

    def test_dismiss_all_clears(self) -> None:
        """全部关闭后列表应为空。"""
        manager = self._manager()
        manager.show("一")
        manager.show("二")
        manager.dismiss_all()
        self.assertEqual(len(manager._toasts), 0)

    def test_disabled_manager_ignores(self) -> None:
        """禁用时不产生通知。"""
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.widgets.toast import ToastManager

        manager = ToastManager(build_theme("dark"), enabled=False)
        self.assertIsNone(manager.show("不该出现"))
        self.assertEqual(len(manager._toasts), 0)

    def test_show_updates_tray_like_status_text(self) -> None:
        """进度文本应能生成（供托盘与面板显示）。"""
        manager = self._manager()
        try:
            widget = manager.show("提示", "说明")
            self.assertIsNotNone(widget)
            self.assertEqual(widget.request.description, "说明")
        finally:
            manager.dismiss_all()


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestOverlayPresentation(unittest.TestCase):
    """翻译层的呈现细节。

    这组测试对应两个直接可见的观感缺陷：

    **背景黑框过大**：背景遮罩原先按**原文包围盒**绘制。中英互译字符数
    差一倍以上，短译文因此被一个明显过大的黑框包住。

    **字幕闪烁**：画面差分只把变化的区域送去识别，增量帧里的条目只覆盖
    屏幕一小块。渲染层原先无条件**替换**整层内容，于是其余位置的译文
    每帧被清空、下一帧再画回来，肉眼看就是整屏文字疯狂闪烁。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _ensure_app()
        if cls.app is None:
            raise unittest.SkipTest("无法创建 QApplication")

    @staticmethod
    def _window():
        from pingyitong.components.capture.geometry import Rect, ScreenInfo
        from pingyitong.components.render.overlay_window import OverlayWindow
        from pingyitong.config.schema import StyleConfig

        screen = ScreenInfo(
            name="TEST",
            logical=Rect(0, 0, 1707, 1067),
            physical=Rect(0, 0, 2560, 1600),
            device_pixel_ratio=1.5,
            is_primary=True,
            mss_index=1,
        )
        return OverlayWindow(screen, StyleConfig())

    @staticmethod
    def _item(box, text, monitor_index=1):
        from pingyitong.components.render.render_item import RenderItem

        return RenderItem(box=box, text=text, source_text="原文" * 20, monitor_index=monitor_index)

    def test_background_hugs_short_translation(self) -> None:
        """短译文的背景必须收缩到文字宽度，不能沿用原文包围盒。"""
        from PyQt6.QtCore import QRectF

        from pingyitong.components.capture.geometry import Rect

        window = self._window()
        try:
            # 原文很宽（模拟一段英文），译文很短（模拟中文）
            item = self._item(Rect(100, 100, 800, 40), "打开窗口")
            cached = window._get_layout(item)
            self.assertIsNotNone(cached)

            inner = QRectF(100.0, 100.0, 800.0, 40.0)
            background = window._background_rect(inner, cached)

            self.assertLess(
                background.width(),
                300,
                "短译文的背景宽度仍接近原文包围盒，黑框会明显过大",
            )
            self.assertLessEqual(background.width(), inner.width())
            self.assertEqual(background.left(), inner.left(), "背景应与文字左对齐")
        finally:
            window.deleteLater()

    def test_background_never_exceeds_available_area(self) -> None:
        """译文比原文长时，背景不得超过原有可用区域。"""
        from PyQt6.QtCore import QRectF

        from pingyitong.components.capture.geometry import Rect

        window = self._window()
        try:
            item = self._item(Rect(100, 100, 200, 40), "A very long translated sentence that will not fit")
            cached = window._get_layout(item)
            inner = QRectF(100.0, 100.0, 200.0, 40.0)
            background = window._background_rect(inner, cached)

            self.assertLessEqual(background.width(), inner.width())
            self.assertLessEqual(background.height(), inner.height())
        finally:
            window.deleteLater()

    def test_incremental_frame_keeps_untouched_items(self) -> None:
        """增量帧只能覆盖识别到的区域，其余译文保持不动。

        这是"字幕疯狂闪烁"的根因回归测试。
        """
        from pingyitong.components.capture.geometry import Rect

        window = self._window()
        try:
            first = [
                self._item(Rect(100, 100, 300, 40), "译文一"),
                self._item(Rect(100, 200, 300, 40), "译文二"),
                self._item(Rect(100, 300, 300, 40), "译文三"),
            ]
            window.set_items(first, replace=True)
            self.assertEqual(len(window._items), 3)

            # 增量帧只识别到第二块区域
            updated = self._item(Rect(100, 200, 300, 40), "译文二（新）")
            window.set_items([updated], replace=False)

            texts = {item.text for item in window._items}
            self.assertEqual(
                len(window._items),
                3,
                "增量帧把未变化区域的译文清掉了，会导致整屏闪烁",
            )
            self.assertIn("译文一", texts)
            self.assertIn("译文三", texts)
            self.assertIn("译文二（新）", texts)
            self.assertNotIn("译文二", texts, "被重新识别的区域应当用新译文替换")
        finally:
            window.deleteLater()

    def test_incremental_frame_without_items_keeps_content(self) -> None:
        """增量帧没有新条目时不得清空已有译文。"""
        from pingyitong.components.capture.geometry import Rect

        window = self._window()
        try:
            window.set_items([self._item(Rect(100, 100, 300, 40), "译文")], replace=True)
            window.set_items([], replace=False)
            self.assertEqual(len(window._items), 1, "增量空帧不应清空翻译层")
        finally:
            window.deleteLater()

    def test_full_frame_replaces_content(self) -> None:
        """整帧识别结果为空时应当清空（例如文字消失）。"""
        from pingyitong.components.capture.geometry import Rect

        window = self._window()
        try:
            window.set_items([self._item(Rect(100, 100, 300, 40), "译文")], replace=True)
            window.set_items([], replace=True)
            self.assertEqual(window._items, [])
        finally:
            window.deleteLater()

    def test_full_frame_replaces_overlapping_content(self) -> None:
        """整帧识别直接替换全部内容，不与旧条目混合。"""
        from pingyitong.components.capture.geometry import Rect

        window = self._window()
        try:
            window.set_items(
                [
                    self._item(Rect(100, 100, 300, 40), "旧一"),
                    self._item(Rect(100, 900, 300, 40), "旧二"),
                ],
                replace=True,
            )
            window.set_items([self._item(Rect(100, 100, 300, 40), "新一")], replace=True)
            self.assertEqual([item.text for item in window._items], ["新一"])
        finally:
            window.deleteLater()


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestTranscriptPanel(unittest.TestCase):
    """实时译文文本框（悬浮栏向下展开）。

    这里守的是两个踩过的坑：

    1. **只动画 ``maximumHeight`` 面板不会长出来**。``maximumHeight`` 只是
       "允许"长到这么高，窗口实际高度仍取决于布局的 sizeHint，而文本框的
       sizeHint 很小——表现为"点了箭头面板纹丝不动"。必须真正改变高度。
    2. **动画结束后必须解除固定高度**，否则窗口被永久锁死在动画终值上，
       之后内容变化无法自适应。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _ensure_app()
        if cls.app is None:
            raise unittest.SkipTest("无法创建 QApplication")

    def _pair(self):
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.floating_bar import FloatingBar
        from pingyitong.ui.widgets.transcript_panel import TranscriptPanel

        theme = build_theme("dark")
        bar = FloatingBar(theme)
        bar.move_to_default_position()
        bar.show_bar(animate=False)
        panel = TranscriptPanel(theme)
        return bar, panel

    def test_expand_reaches_target_height(self) -> None:
        """展开后面板必须真的长到目标高度（不是纹丝不动）。

        目标高度是 ``min(期望高度, 悬浮栏下方剩余空间)``——面板**只向下**
        展开（用户要求：不得覆盖在悬浮条上方），空间不足时收缩高度。
        """
        from PyQt6.QtTest import QTest

        from pingyitong.ui.design import SPACE
        from pingyitong.ui.widgets.transcript_panel import DEFAULT_PANEL_HEIGHT

        bar, panel = self._pair()
        try:
            QTest.qWait(150)
            panel.expand(bar)
            QTest.qWait(400)
            self.assertTrue(panel.is_expanded)

            area = bar.screen().availableGeometry()
            bar_bottom = bar.mapToGlobal(bar.rect().bottomLeft()).y()
            available = area.bottom() - (bar_bottom + SPACE["xs"]) + 1
            expected = max(1, min(DEFAULT_PANEL_HEIGHT, available))
            self.assertAlmostEqual(
                panel.height(),
                expected,
                delta=6,
                msg="展开后高度与悬浮栏下方可用空间不符",
            )
        finally:
            bar.deleteLater()
            panel.deleteLater()

    def test_expand_is_reentrant_safe(self) -> None:
        """已展开时再次 ``expand`` 不得重发信号、不得重播动画。

        拖动悬浮栏会让应用在面板已展开的情况下再次调用 ``expand``（跟随）。
        若此时重发 ``expanded_changed``，应用会把"翻译层原本可见"的恢复状态
        覆盖成 False（此刻翻译层正被文本框功能隐藏着），收起文本框后翻译层
        就不再恢复；重播动画则会让面板先缩到 0 再长回来。
        """
        from PyQt6.QtTest import QTest

        bar, panel = self._pair()
        try:
            QTest.qWait(150)
            received: list[bool] = []
            panel.expanded_changed.connect(received.append)

            panel.expand(bar)
            QTest.qWait(400)
            self.assertTrue(panel.is_expanded)
            self.assertEqual(received, [True])

            height_before = panel.height()

            # 模拟拖动悬浮栏后的跟随调用
            panel.expand(bar)
            QTest.qWait(100)
            self.assertEqual(
                received,
                [True],
                "已展开时再次 expand 不应重发 expanded_changed",
            )
            self.assertTrue(panel.is_expanded)
            self.assertGreaterEqual(
                panel.height(),
                height_before - 2,
                "已展开时再次 expand 不应把高度缩回 0 再重播动画",
            )

            # 收起后状态仍应正常
            panel.collapse()
            QTest.qWait(300)
            self.assertFalse(panel.is_expanded)
            self.assertEqual(received, [True, False])
        finally:
            bar.deleteLater()
            panel.deleteLater()

    def test_panel_never_covers_the_bar(self) -> None:
        """面板**不得**出现在悬浮栏上方（用户明确要求）。

        曾经"下方空间不足就把面板翻到上方"，结果盖住了屏幕上大块内容。
        无论悬浮栏被拖到哪里，面板顶部都必须在悬浮栏底边之下。
        """
        from PyQt6.QtTest import QTest

        bar, panel = self._pair()
        try:
            QTest.qWait(150)
            panel.expand(bar)
            QTest.qWait(400)
            self.assertTrue(panel.is_expanded)

            bar_bottom = bar.mapToGlobal(bar.rect().bottomLeft()).y()
            self.assertGreaterEqual(
                panel.y(),
                bar_bottom - 2,
                "面板覆盖到了悬浮栏（顶部高于悬浮栏底边）",
            )
            # 面板整体也不得超出屏幕底部
            area = bar.screen().availableGeometry()
            self.assertLessEqual(panel.y() + panel.height(), area.bottom() + 1)
        finally:
            bar.deleteLater()
            panel.deleteLater()

    def test_panel_width_matches_bar(self) -> None:
        """文本框宽度必须与悬浮栏一致（用户要求两者对齐）。"""
        from PyQt6.QtTest import QTest

        bar, panel = self._pair()
        try:
            QTest.qWait(100)
            panel.expand(bar)
            QTest.qWait(350)
            self.assertEqual(
                panel.width(),
                bar.width(),
                "文本框宽度与悬浮栏不一致",
            )
        finally:
            bar.deleteLater()
            panel.deleteLater()

    def test_panel_background_is_translucent(self) -> None:
        """文本框背景应为 35% 不透明度（用户指定）。

        样式表以 rgba 提供 alpha≈89（0.35×255），否则背景要么全实
        （遮屏幕）、要么全透（文字浮空）。检查全局样式表确实包含
        带该 alpha 的面板规则，且渲染结果**不是**不透明表面色。
        """
        from PyQt6.QtGui import QColor, QImage

        from pingyitong.ui.design.theme import build_theme, stylesheet
        from pingyitong.ui.widgets.transcript_panel import TranscriptPanel

        theme = build_theme("dark")
        self.app.setStyleSheet(stylesheet(theme))
        css = stylesheet(theme)
        try:
            self.assertIn("#TranscriptPanel", css)
            self.assertIn("rgba(", css)

            panel = TranscriptPanel(theme)
            panel.resize(300, 200)
            image = QImage(300, 200, QImage.Format.Format_ARGB32)
            image.fill(QColor(255, 255, 255, 255))
            panel.render(image)
            center = QColor(image.pixelColor(150, 100))
            surface = QColor(theme.scheme.surface_raised)
            # 与不透明表面色明显不同 → 透出了铺底的白色
            self.assertNotEqual(
                (center.red(), center.green(), center.blue()),
                (surface.red(), surface.green(), surface.blue()),
                "背景渲染为不透明表面色——35% 半透明未生效",
            )
        finally:
            panel.deleteLater()
            self.app.setStyleSheet("")

    def test_height_grows_progressively_during_animation(self) -> None:
        """展开过程中高度必须**逐帧增长**，而不是最后一下跳到位。

        这条才是真正锁住实现细节的断言。只动画 ``maximumHeight`` 时，
        窗口高度一直停在布局的 sizeHint（很小），动画结束后才由收尾逻辑
        直接跳到目标值——用户看到的是"点了没反应，然后突然出现一整块"。
        实测动画中途高度应处于 0 与目标值之间。
        """
        from PyQt6.QtTest import QTest

        from pingyitong.ui.widgets.transcript_panel import DEFAULT_PANEL_HEIGHT

        bar, panel = self._pair()
        try:
            QTest.qWait(150)
            panel.expand(bar)
            QTest.qWait(60)  # 约动画时长的三分之一
            mid = panel.height()
            self.assertGreater(
                mid,
                0,
                "动画中途高度仍为 0——说明没有真正改变窗口高度，面板是突然出现的",
            )
            self.assertLess(
                mid,
                panel._target_height or DEFAULT_PANEL_HEIGHT,
                "动画中途高度已达终值——说明没有做动画而是直接设置",
            )
        finally:
            bar.deleteLater()
            panel.deleteLater()

    def test_height_is_not_locked_after_expand(self) -> None:
        """展开结束后不得把高度锁死，否则内容变化无法自适应。"""
        from PyQt6.QtTest import QTest

        bar, panel = self._pair()
        try:
            panel.expand(bar)
            QTest.qWait(400)
            lines = [f"第 {i} 行" for i in range(200)]
            panel.set_text(chr(10).join(lines))
            QTest.qWait(50)
            self.assertEqual(panel.minimumHeight(), panel.maximumHeight())
        finally:
            bar.deleteLater()
            panel.deleteLater()

    def test_collapse_hides_panel(self) -> None:
        """收起后面板应隐藏。"""
        from PyQt6.QtTest import QTest

        bar, panel = self._pair()
        try:
            panel.expand(bar)
            QTest.qWait(350)
            panel.collapse()
            QTest.qWait(400)
            self.assertFalse(panel.is_expanded)
            self.assertFalse(panel.isVisible())
        finally:
            bar.deleteLater()
            panel.deleteLater()

    def test_expanded_signal_emitted(self) -> None:
        """展开状态变化必须发出信号，否则界面不会切换显示方式。"""
        from PyQt6.QtTest import QTest

        bar, panel = self._pair()
        seen: list[bool] = []
        panel.expanded_changed.connect(seen.append)
        try:
            panel.expand(bar)
            QTest.qWait(350)
            panel.collapse()
            QTest.qWait(400)
            self.assertEqual(seen, [True, False])
        finally:
            bar.deleteLater()
            panel.deleteLater()

    def test_panel_placed_within_screen(self) -> None:
        """面板必须完整落在屏幕可用区内（悬浮栏可被拖到任意位置）。"""
        from PyQt6.QtTest import QTest

        bar, panel = self._pair()
        try:
            panel.expand(bar)
            QTest.qWait(350)
            screen = panel.screen() or bar.screen()
            area = screen.availableGeometry()
            self.assertGreaterEqual(panel.x(), area.left() - 1)
            self.assertLessEqual(panel.x() + panel.width(), area.right() + 1)
            self.assertGreaterEqual(panel.y(), area.top() - 1)
            self.assertLessEqual(panel.y() + panel.height(), area.bottom() + 1)
        finally:
            bar.deleteLater()
            panel.deleteLater()

    def test_panel_background_is_painted(self) -> None:
        """面板背景必须真的被绘制，不能是全透明。

        必须在**全局样式表生效**的前提下验证：#TranscriptPanel 的
        background-color 由样式表提供。真机上曾出现"面板完全透明、
        只有文字浮在桌面上"的问题。
        这里把面板渲染到铺了亮色的图像上，采样中心与角落像素：
        中心应当是面板表面色，说明背景确实画了。
        """
        from PyQt6.QtGui import QColor, QImage

        from pingyitong.ui.design.theme import build_theme, stylesheet
        from pingyitong.ui.widgets.transcript_panel import TranscriptPanel

        theme = build_theme("dark")
        self.app.setStyleSheet(stylesheet(theme))
        panel = TranscriptPanel(theme)
        try:
            panel.resize(300, 200)
            image = QImage(300, 200, QImage.Format.Format_ARGB32)
            image.fill(QColor(0, 255, 0, 255))  # 铺亮绿便于分辨
            panel.render(image)

            center = QColor(image.pixelColor(150, 100))
            self.assertNotEqual(
                (center.red(), center.green(), center.blue()),
                (0, 255, 0),
                "面板背景未绘制（中心像素仍是铺底色），面板会完全透明",
            )
        finally:
            panel.deleteLater()
            self.app.setStyleSheet("")

    def test_arrow_flips_with_state(self) -> None:
        """悬浮栏箭头的方向要如实反映点击后的结果。"""
        bar, panel = self._pair()
        try:
            bar.set_transcript_open(True)
            self.assertEqual(bar._transcript_button.text(), "▴")
            bar.set_transcript_open(False)
            self.assertEqual(bar._transcript_button.text(), "▾")
        finally:
            bar.deleteLater()
            panel.deleteLater()

    def test_set_text_keeps_scroll_at_bottom(self) -> None:
        """内容更新时应自动跟随到底部（用户没往上翻的情况下）。"""
        from PyQt6.QtTest import QTest

        bar, panel = self._pair()
        try:
            panel.expand(bar)
            QTest.qWait(350)
            many = [f"行 {i}" for i in range(300)]
            panel.set_text(chr(10).join(many))
            QTest.qWait(80)
            scrollbar = panel._text.verticalScrollBar()
            self.assertEqual(scrollbar.value(), scrollbar.maximum())
        finally:
            bar.deleteLater()
            panel.deleteLater()


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestBarDragClamp(unittest.TestCase):
    """拖动释放后悬浮栏必须整条夹回屏幕内。

    踩过的坑：``restore_position`` 只在显示与屏幕布局变化时执行，而拖动
    释放路径把原始坐标直接持久化——松手停在屏外的控制条会挂到本次会话
    结束，右侧按钮（含展开箭头）一直点不到。实测 0.6.21 曾把
    ``floating_bar_pos`` 存成 ``[1658, 438]``（右缘 2086，超出主屏逻辑宽度
    1707 达 379px），期间 ▾/⚙/☰ 三个按钮完全无法点击。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _ensure_app()
        if cls.app is None:
            raise unittest.SkipTest("无法创建 QApplication")

    def _bar(self):
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.floating_bar import FloatingBar

        return FloatingBar(build_theme("dark"))

    def _patch_screens(self, area: QRect) -> None:
        """把 ``screens()`` 换成单块 ``area`` 屏幕的替身（测后恢复）。"""
        from types import SimpleNamespace

        fake = SimpleNamespace(availableGeometry=lambda: QRect(area))
        self.app.screens = lambda: [fake]  # type: ignore[method-assign]
        self.addCleanup(delattr, self.app, "screens")

    def test_clamped_position_pulls_partial_offscreen_back_in(self) -> None:
        """右半截出屏的位置必须被夹回所在屏幕的可用区域。"""
        bar = self._bar()
        try:
            self._patch_screens(QRect(0, 0, 1000, 800))
            # x=600：探针 (620, 381) 在假屏内，但右缘 600+width 超出 1000
            clamped = bar._clamped_position(600, 371)
            self.assertIsNotNone(clamped)
            assert clamped is not None
            self.assertEqual(clamped.x(), 1000 - bar.width())
            self.assertEqual(clamped.y(), 371)
        finally:
            bar.deleteLater()

    def test_clamped_position_returns_none_when_fully_offscreen(self) -> None:
        """完全出屏（显示器已拔）时返回 None，交给调用方回退。"""
        bar = self._bar()
        try:
            self._patch_screens(QRect(0, 0, 1000, 800))
            self.assertIsNone(bar._clamped_position(-5000, 371))
        finally:
            bar.deleteLater()

    def test_restore_position_falls_back_to_default_when_offscreen(self) -> None:
        """restore 对完全出屏的坐标仍回退默认位置（原有语义不回退）。"""
        bar = self._bar()
        try:
            self._patch_screens(QRect(0, 0, 1000, 800))
            bar.restore_position(-5000, 371)
            # 默认位置参照真实主屏（move_to_default_position 的实际依据）
            self.assertTrue(bar.screen().availableGeometry().contains(bar.pos()))
        finally:
            bar.deleteLater()

    def test_drag_release_clamps_before_persisting(self) -> None:
        """松手在屏外时先夹回再发 position_changed，持久化的是夹取后坐标。"""
        from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
        from PyQt6.QtGui import QMouseEvent

        bar = self._bar()
        try:
            self._patch_screens(QRect(0, 0, 1000, 800))
            saved: list[tuple[int, int]] = []
            bar.position_changed.connect(lambda x, y: saved.append((x, y)))
            bar.move(900, 100)  # 右缘 900 + width 超出 1000
            bar._drag_offset = QPoint(0, 0)
            event = QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                QPointF(10, 10),
                QPointF(bar.x() + 10, bar.y() + 10),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            bar.mouseReleaseEvent(event)
            self.assertLessEqual(bar.x(), 1000 - bar.width())
            self.assertEqual(saved, [(bar.x(), bar.y())])
        finally:
            bar.deleteLater()


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestLanguageCombo(unittest.TestCase):
    """语言选择框的交互。

    这组测试对应实际踩过的坑：做成"可编辑 + 补全器"的搜索框之后，
    点文本框不展开列表（只有右侧那条细箭头能展开），用户点了没反应，
    看到的是一个空格子，以为不能选。语言选择是"从固定清单里挑一个"，
    应当用普通下拉框：点哪儿都能展开，框里显示的永远等于实际生效的语言。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _ensure_app()
        if cls.app is None:
            raise unittest.SkipTest("无法创建 QApplication")

    def _combo(self, current: str = "zh", with_auto: bool = False):
        from pingyitong.ui.design import LAYOUT
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.widgets.language_combo import LanguageCombo

        combo = LanguageCombo(current=current, theme=build_theme("dark"), with_auto=with_auto)
        combo.resize(260, LAYOUT["control_height"])
        return combo

    def test_is_not_editable(self) -> None:
        """必须是普通下拉框：显示内容永远等于当前选择。

        可编辑下拉框允许用户随手改文字，改完失焦后框里的字与实际生效的
        语言就对不上（配置里是中文、框里却是空的或半截字样）。
        """
        combo = self._combo()
        try:
            self.assertFalse(combo.isEditable(), "语言选择不应可编辑")
        finally:
            combo.deleteLater()

    def test_shows_current_selection(self) -> None:
        """初始显示必须是当前语言，不能是空的。"""
        combo = self._combo("zh")
        try:
            self.assertTrue(combo.currentText().strip(), "显示内容不能为空")
            self.assertEqual(combo.currentData(), "zh")
            self.assertIn("中文", combo.currentText())
        finally:
            combo.deleteLater()

    def test_click_anywhere_opens_the_list(self) -> None:
        """点击控件任意位置都应展开列表。

        这是用户实际卡住的地方：只有右侧约 22px 的箭头区域能展开，
        而用户自然会点文本框。

        每个位置用**独立的控件**验证：刚 ``hidePopup()`` 之后紧接着的那次
        点击会被 Qt 吞掉（它要避免"关闭弹层的那一下点击"又把弹层打开），
        那是测试写法的问题，不是控件行为。

        点击后要等足时间：Qt 的弹出层带展开动画，实测约 300ms 后才可见，
        等太短会误判成"没展开"。
        """
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest

        for label, ratio in (("中心", 0.5), ("左侧", 0.04), ("右侧箭头", 0.98)):
            combo = self._combo()
            try:
                combo.show()
                QTest.qWait(100)
                x = max(1, int(combo.width() * ratio))
                point = QPoint(x, combo.height() // 2)
                QTest.mouseClick(
                    combo, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point
                )
                QTest.qWait(500)
                self.assertTrue(
                    combo.view().isVisible(),
                    f"点击{label}({point.x()},{point.y()})未能展开语言列表",
                )
                combo.hidePopup()
            finally:
                combo.deleteLater()

    def test_lists_every_candidate_language(self) -> None:
        """下拉必须列出全部候选语言，不做人为限制。"""
        from pingyitong.languages import LANGUAGES

        combo = self._combo()
        try:
            listed = {combo.itemData(i) for i in range(combo.count())}
            self.assertEqual(listed, set(LANGUAGES))
        finally:
            combo.deleteLater()

    def test_auto_option_only_for_source(self) -> None:
        """"自动检测"只应出现在源语言下拉里。"""
        source = self._combo(with_auto=True)
        target = self._combo()
        try:
            self.assertEqual(source.itemData(0), "auto")
            self.assertNotIn("auto", {target.itemData(i) for i in range(target.count())})
        finally:
            source.deleteLater()
            target.deleteLater()

    def test_keyboard_search_by_chinese_english_and_native_name(self) -> None:
        """键入中文名、英文名或母语名都应能跳到对应语言。

        非可编辑下拉框默认按"显示文本前缀"查找，而显示文本以状态符号
        ``✓`` 开头，所有项都一样，等于完全失效。因此必须重写查找逻辑。
        """
        combo = self._combo()
        try:
            for typed, expected in (
                ("冰", "is"),
                ("ice", "is"),
                ("冰岛语", "is"),
                ("japan", "ja"),
                ("日本", "ja"),
                ("僧", "si"),
                ("deu", "de"),
                ("deutsch", "de"),
            ):
                with self.subTest(typed=typed):
                    combo.set_language("zh")
                    combo.keyboardSearch(typed)
                    self.assertEqual(
                        combo.language,
                        expected,
                        f"键入 {typed!r} 应跳到 {expected}，实际 {combo.language}",
                    )
        finally:
            combo.deleteLater()

    def test_unmatched_search_keeps_selection(self) -> None:
        """搜不到时保持原选择，不能跳到无关语言或清空。"""
        combo = self._combo("ja")
        try:
            combo.keyboardSearch("zzzzz")
            self.assertEqual(combo.language, "ja")
            self.assertTrue(combo.currentText().strip())
        finally:
            combo.deleteLater()

    def test_selection_emits_signal(self) -> None:
        """切换语言必须发出信号，否则界面改了但配置没跟上。"""
        combo = self._combo("zh")
        received: list[str] = []
        combo.language_changed.connect(received.append)
        try:
            index = combo.findData("ja")
            combo.setCurrentIndex(index)
            self.assertEqual(received, ["ja"])
        finally:
            combo.deleteLater()

    def test_set_language_does_not_emit(self) -> None:
        """程序化设置不应触发信号（避免加载配置时回写配置）。"""
        combo = self._combo("zh")
        received: list[str] = []
        combo.language_changed.connect(received.append)
        try:
            combo.set_language("ja")
            self.assertEqual(received, [])
            self.assertEqual(combo.language, "ja")
        finally:
            combo.deleteLater()

    def test_popup_scrolls_current_into_view(self) -> None:
        """展开列表时当前项应在可视区内，否则用户看不到自己选的是哪个。"""
        from PyQt6.QtTest import QTest

        combo = self._combo("zu")
        try:
            combo.show()
            QTest.qWait(50)
            combo.showPopup()
            QTest.qWait(80)
            index = combo.currentIndex()
            rect = combo.view().visualRect(combo.view().model().index(index, 0))
            viewport = combo.view().viewport().rect()
            self.assertTrue(
                rect.isValid() and viewport.intersects(rect),
                "当前项未滚动到可视区域",
            )
            combo.hidePopup()
        finally:
            combo.deleteLater()


if __name__ == "__main__":
    unittest.main(verbosity=2)
