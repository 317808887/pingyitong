"""全局热键解析测试。

热键字符串由用户在配置文件中手工编辑，格式错误的可能性很高。
解析器必须给出明确错误而不是静默失效——否则用户会困惑于
"为什么我设的热键没反应"。

注意：本测试只覆盖**字符串解析**这一纯逻辑部分，
不涉及 ``RegisterHotKey`` 系统调用（那需要真实的消息循环）。
"""

from __future__ import annotations

import unittest

from pingyitong.services.hotkey_service import (
    ACTION_COPY_RESULT,
    ACTION_LABELS,
    ACTION_SWITCH_MODEL,
    ACTION_TOGGLE_OVERLAY,
    ACTION_TOGGLE_TRANSLATION,
    HotkeyError,
    _fallback_sequence,
    default_bindings,
    format_sequence,
    parse_sequence,
)


class TestParseSequence(unittest.TestCase):
    """热键字符串解析。"""

    def test_ctrl_alt_letter(self) -> None:
        """典型组合 Ctrl+Alt+T。"""
        modifiers, vk = parse_sequence("Ctrl+Alt+T")
        self.assertEqual(vk, ord("T"))
        self.assertTrue(modifiers & 0x0002)  # MOD_CONTROL
        self.assertTrue(modifiers & 0x0001)  # MOD_ALT

    def test_case_insensitive(self) -> None:
        """键名大小写不敏感（用户可能写成 ctrl+alt+t）。"""
        self.assertEqual(parse_sequence("ctrl+alt+t"), parse_sequence("CTRL+ALT+T"))

    def test_function_keys(self) -> None:
        """功能键 F1-F24。"""
        self.assertEqual(parse_sequence("F1")[1], 0x70)
        self.assertEqual(parse_sequence("F12")[1], 0x7B)
        self.assertEqual(parse_sequence("F24")[1], 0x87)

    def test_digit_keys(self) -> None:
        """数字键。"""
        self.assertEqual(parse_sequence("Ctrl+1")[1], ord("1"))

    def test_named_keys(self) -> None:
        """具名特殊键。"""
        self.assertEqual(parse_sequence("Ctrl+Space")[1], 0x20)
        self.assertEqual(parse_sequence("Alt+Enter")[1], 0x0D)
        self.assertEqual(parse_sequence("Ctrl+Shift+Esc")[1], 0x1B)

    def test_win_modifier(self) -> None:
        """Win 键修饰符。"""
        modifiers, _ = parse_sequence("Win+Shift+S")
        self.assertTrue(modifiers & 0x0008)  # MOD_WIN
        self.assertTrue(modifiers & 0x0004)  # MOD_SHIFT

    def test_meta_aliases(self) -> None:
        """meta / super 是 Win 的别名。"""
        self.assertEqual(parse_sequence("Meta+A")[0], parse_sequence("Win+A")[0])
        self.assertEqual(parse_sequence("Super+A")[0], parse_sequence("Win+A")[0])

    def test_single_key_without_modifier(self) -> None:
        """允许单键（虽然实践中不推荐）。"""
        modifiers, vk = parse_sequence("F9")
        self.assertEqual(modifiers, 0)
        self.assertEqual(vk, 0x78)

    def test_whitespace_tolerated(self) -> None:
        """键名两侧的空格应被忽略。"""
        self.assertEqual(parse_sequence(" Ctrl + Alt + T "), parse_sequence("Ctrl+Alt+T"))

    def test_empty_raises(self) -> None:
        """空字符串应报错而不是返回无效热键。"""
        with self.assertRaises(HotkeyError):
            parse_sequence("")

    def test_whitespace_only_raises(self) -> None:
        """纯空白同样报错。"""
        with self.assertRaises(HotkeyError):
            parse_sequence("   ")

    def test_modifier_only_raises(self) -> None:
        """只有修饰键没有主键时应报错。

        否则会注册出一个能拦截全部 Ctrl 按键的无效热键。
        """
        with self.assertRaises(HotkeyError):
            parse_sequence("Ctrl+Alt")

    def test_multiple_main_keys_raises(self) -> None:
        """多个主键应报错（含混配置多半是写错了）。"""
        with self.assertRaises(HotkeyError):
            parse_sequence("Ctrl+A+B")

    def test_unknown_key_raises(self) -> None:
        """无法识别的键名应报错并指明是哪个键。"""
        with self.assertRaises(HotkeyError) as context:
            parse_sequence("Ctrl+这不是键")
        self.assertIn("无法识别", str(context.exception))

    def test_out_of_range_function_key_raises(self) -> None:
        """F25 超出范围应报错。"""
        with self.assertRaises(HotkeyError):
            parse_sequence("Ctrl+F25")

    def test_non_ascii_keys_raise_hotkey_error_without_crashing_thread(self) -> None:
        for sequence in ("Ctrl+ß", "Ctrl+中", "Ctrl+F²", "Ctrl+１"):
            with self.subTest(sequence=sequence), self.assertRaises(HotkeyError):
                parse_sequence(sequence)

    def test_literal_punctuation_keys(self) -> None:
        self.assertEqual(parse_sequence("Ctrl+.")[1], 0xBE)
        self.assertEqual(parse_sequence("Ctrl+/")[1], 0xBF)
        self.assertEqual(parse_sequence("Ctrl++")[1], 0xBB)
        self.assertEqual(parse_sequence("+")[1], 0xBB)


class TestFormatSequence(unittest.TestCase):
    """热键显示格式化。"""

    def test_capitalises_modifiers(self) -> None:
        """修饰键名规范为大写缩写。"""
        self.assertEqual(format_sequence("ctrl+alt+t"), "Ctrl+Alt+T")

    def test_control_alias(self) -> None:
        """control 显示为 Ctrl。"""
        self.assertEqual(format_sequence("control+m"), "Ctrl+M")

    def test_meta_alias(self) -> None:
        """meta / super 显示为 Win。"""
        self.assertEqual(format_sequence("meta+k"), "Win+K")
        self.assertEqual(format_sequence("super+k"), "Win+K")

    def test_named_key_capitalised(self) -> None:
        """具名键首字母大写。"""
        self.assertEqual(format_sequence("ctrl+space"), "Ctrl+Space")

    def test_empty_input(self) -> None:
        """空输入原样返回。"""
        self.assertEqual(format_sequence(""), "")


class TestDefaultBindings(unittest.TestCase):
    """默认热键绑定。"""

    def test_bindings_from_config(self) -> None:
        """应从配置节提取四个动作的绑定。"""

        class Config:
            toggle_translation = "Ctrl+Alt+T"
            switch_model = "Ctrl+Alt+M"
            copy_result = "Ctrl+Alt+C"
            toggle_overlay = "Ctrl+Alt+H"

        bindings = default_bindings(Config())
        self.assertEqual(bindings[ACTION_TOGGLE_TRANSLATION], "Ctrl+Alt+T")
        self.assertEqual(bindings[ACTION_SWITCH_MODEL], "Ctrl+Alt+M")
        self.assertEqual(bindings[ACTION_COPY_RESULT], "Ctrl+Alt+C")
        self.assertEqual(bindings[ACTION_TOGGLE_OVERLAY], "Ctrl+Alt+H")

    def test_defaults_are_parseable(self) -> None:
        """默认热键必须能被解析——否则开箱即用就是坏的。"""

        class Config:
            pass

        for sequence in default_bindings(Config()).values():
            with self.subTest(sequence=sequence):
                modifiers, vk = parse_sequence(sequence)
                self.assertNotEqual(vk, 0)

    def test_all_actions_have_labels(self) -> None:
        """每个动作都应有可展示的中文名称（设置面板与日志会用到）。"""

        class Config:
            pass

        for action in default_bindings(Config()):
            with self.subTest(action=action):
                self.assertIn(action, ACTION_LABELS)
                self.assertTrue(ACTION_LABELS[action])

    def test_default_model_shortcut_has_non_conflicting_fallback(self) -> None:
        desired = {
            ACTION_SWITCH_MODEL: "Ctrl+Alt+M",
            ACTION_COPY_RESULT: "Ctrl+Alt+C",
        }
        self.assertEqual(
            _fallback_sequence(ACTION_SWITCH_MODEL, "Ctrl+Alt+M", desired),
            "Ctrl+Alt+F1",
        )
        desired[ACTION_COPY_RESULT] = "Ctrl+Alt+F1"
        self.assertEqual(_fallback_sequence(ACTION_SWITCH_MODEL, "Ctrl+Alt+M", desired), "")
        self.assertEqual(_fallback_sequence(ACTION_SWITCH_MODEL, "Ctrl+Alt+F2", desired), "")


class TestActionConstants(unittest.TestCase):
    """热键动作常量。"""

    def test_action_constants_distinct(self) -> None:
        """四个动作标识必须互不相同，否则会互相覆盖。"""
        actions = {
            ACTION_TOGGLE_TRANSLATION,
            ACTION_SWITCH_MODEL,
            ACTION_COPY_RESULT,
            ACTION_TOGGLE_OVERLAY,
        }
        self.assertEqual(len(actions), 4)


class TestHotkeyDispatch(unittest.TestCase):
    """热键触发后的分发逻辑。

    真实的全局热键（``RegisterHotKey`` + 消息循环）已在真机上手工验证通过：
    注册成功、合成按键能触发回调。这里覆盖的是**消息循环收到 WM_HOTKEY 之后**
    的分发逻辑——即"系统热键 ID → 动作 → 回调"的映射。

    测试直接调用**真实的** ``_HotkeyThread._dispatch``（只是不启动消息循环），
    而不是用替身：用替身测分发逻辑等于什么都没测。
    """

    def _make_thread(self, bindings: list) -> tuple:
        from pingyitong.services.hotkey_service import HotkeyService, _HotkeyThread

        fired: list[str] = []
        service = HotkeyService(on_action=fired.append)
        thread = _HotkeyThread(service)
        # 注入绑定表，但不启动线程（不注册真实热键、不进入消息循环）
        thread._bindings = list(bindings)
        return thread, fired

    def _binding(self, action: str, hotkey_id: int, registered: bool = True, error: str = ""):
        from pingyitong.services.hotkey_service import HotkeyBinding

        return HotkeyBinding(
            action=action,
            sequence="Ctrl+Alt+F1",
            hotkey_id=hotkey_id,
            registered=registered,
            error=error,
        )

    def test_dispatch_routes_id_to_action(self) -> None:
        """系统热键 ID 应被正确映射回对应动作。"""
        thread, fired = self._make_thread(
            [
                self._binding(ACTION_TOGGLE_TRANSLATION, 1),
                self._binding(ACTION_SWITCH_MODEL, 2),
            ]
        )

        thread._dispatch(2)
        self.assertEqual(fired, [ACTION_SWITCH_MODEL])

        thread._dispatch(1)
        self.assertEqual(fired, [ACTION_SWITCH_MODEL, ACTION_TOGGLE_TRANSLATION])

    def test_dispatch_ignores_unknown_id(self) -> None:
        """未知 ID 不应触发任何动作。"""
        thread, fired = self._make_thread([self._binding(ACTION_TOGGLE_TRANSLATION, 1)])
        thread._dispatch(999)
        self.assertEqual(fired, [])

    def test_dispatch_skips_unregistered_binding(self) -> None:
        """注册失败的绑定即使 ID 相同也不应触发。

        否则用户会误以为热键生效了，而实际按下时触发的是别的软件。
        """
        thread, fired = self._make_thread(
            [self._binding(ACTION_TOGGLE_TRANSLATION, 1, registered=False, error="被占用")]
        )
        thread._dispatch(1)
        self.assertEqual(fired, [])

    def test_callback_exception_does_not_propagate(self) -> None:
        """回调抛异常不应让消息循环崩溃。

        回调来自界面层，界面出问题不该导致热键整体失效。
        """
        from pingyitong.services.hotkey_service import HotkeyService, _HotkeyThread

        def boom(action: str) -> None:
            raise RuntimeError("界面炸了")

        service = HotkeyService(on_action=boom)
        thread = _HotkeyThread(service)
        thread._bindings = [self._binding(ACTION_TOGGLE_TRANSLATION, 1)]
        # 不应抛异常
        thread._dispatch(1)

    def test_dispatch_with_no_bindings_is_safe(self) -> None:
        """没有绑定时分发应安全返回。"""
        thread, fired = self._make_thread([])
        thread._dispatch(1)
        self.assertEqual(fired, [])

    def test_trigger_ignores_unknown_action(self) -> None:
        """未知动作应被忽略而不是抛异常。"""
        from pingyitong.services.hotkey_service import HotkeyService

        fired: list[str] = []
        service = HotkeyService(on_action=fired.append)
        service.trigger("不存在的动作")
        self.assertEqual(fired, [])

    def test_trigger_known_action(self) -> None:
        """已知动作应回调。"""
        from pingyitong.services.hotkey_service import HotkeyService

        fired: list[str] = []
        service = HotkeyService(on_action=fired.append)
        service.trigger(ACTION_COPY_RESULT)
        self.assertEqual(fired, [ACTION_COPY_RESULT])

    def test_set_callback_replaces(self) -> None:
        """替换回调后应由新回调接管。"""
        from pingyitong.services.hotkey_service import HotkeyService

        fired: list[str] = []
        service = HotkeyService(on_action=fired.append)
        second: list[str] = []
        service.set_callback(second.append)
        service.trigger(ACTION_SWITCH_MODEL)
        self.assertEqual(fired, [])
        self.assertEqual(second, [ACTION_SWITCH_MODEL])

    def test_bindings_empty_before_start(self) -> None:
        """未启动时不应报告任何绑定。"""
        from pingyitong.services.hotkey_service import HotkeyService

        self.assertEqual(HotkeyService().bindings(), [])

    def test_is_active_false_before_start(self) -> None:
        """未启动时不应报告为活动状态。"""
        from pingyitong.services.hotkey_service import HotkeyService

        self.assertFalse(HotkeyService().is_active)

    def test_disabled_service_does_not_start(self) -> None:
        """禁用后不应处于活动状态。"""
        from pingyitong.services.hotkey_service import HotkeyService

        service = HotkeyService()
        service.set_enabled(False)
        self.assertFalse(service.is_active)

    def test_default_bindings_are_registrable(self) -> None:
        """默认热键必须都能被解析（否则开箱即用就是坏的）。"""
        from pingyitong.services.hotkey_service import default_bindings

        class Config:
            toggle_translation = "Ctrl+Alt+F1"
            switch_model = "Ctrl+Alt+F2"
            copy_result = "Ctrl+Alt+F3"
            toggle_overlay = "Ctrl+Alt+F4"

        for action, sequence in default_bindings(Config()).items():
            with self.subTest(action=action):
                modifiers, vk = parse_sequence(sequence)
                self.assertNotEqual(vk, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
