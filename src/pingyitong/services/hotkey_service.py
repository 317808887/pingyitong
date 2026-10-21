"""全局热键管理（自研）。

对应产品功能规划三·（二）：用户在任意应用中均可通过全局热键
一键开启/暂停翻译、切换模型、复制译文、隐藏翻译层。

实现方式
    直接调用 Windows 的 ``RegisterHotKey`` / ``UnregisterHotKey`` API，
    并在独立线程中运行 ``GetMessageW`` 消息循环接收 ``WM_HOTKEY``。
    不引入 ``keyboard``、``pynput`` 等第三方库，原因：

    - 那些库普遍通过全局键盘钩子实现，会被安全软件拦截，也会带来
      不必要的输入监听风险——对一个"隐私优先"的离线工具不可接受；
    - ``RegisterHotKey`` 是系统级注册，不监听按键内容，隐私边界清晰。

线程模型
    消息循环必须与 ``RegisterHotKey`` 在同一线程，因此热键注册与
    消息泵都放在 :class:`_HotkeyThread` 中。触发后通过回调转交上层，
    由上层用 Qt 信号切回主线程（Qt 界面只能在主线程操作）。
"""

from __future__ import annotations

import ctypes
import sys
import threading
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)

_IS_WINDOWS = sys.platform == "win32"

# Windows 常量
_WM_HOTKEY = 0x0312
_MOD_ALT = 0x0001
_MOD_CONTROL = 0x0002
_MOD_SHIFT = 0x0004
_MOD_WIN = 0x0008
_MOD_NOREPEAT = 0x4000

#: 修饰键名称 -> Windows 修饰位
_MODIFIERS: dict[str, int] = {
    "ctrl": _MOD_CONTROL,
    "control": _MOD_CONTROL,
    "alt": _MOD_ALT,
    "shift": _MOD_SHIFT,
    "win": _MOD_WIN,
    "meta": _MOD_WIN,
    "super": _MOD_WIN,
}

#: 特殊键名称 -> 虚拟键码
_SPECIAL_KEYS: dict[str, int] = {
    "space": 0x20,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "esc": 0x1B,
    "escape": 0x1B,
    "backspace": 0x08,
    "delete": 0x2E,
    "del": 0x2E,
    "insert": 0x2D,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "plus": 0xBB,
    "minus": 0xBD,
    "comma": 0xBC,
    "period": 0xBE,
    "slash": 0xBF,
    "semicolon": 0xBA,
    "quote": 0xDE,
    "bracketleft": 0xDB,
    "bracketright": 0xDD,
    "backslash": 0xDC,
    "grave": 0xC0,
    ".": 0xBE,
    "-": 0xBD,
    ",": 0xBC,
    "/": 0xBF,
    ";": 0xBA,
    "'": 0xDE,
    "[": 0xDB,
    "]": 0xDD,
    "\\": 0xDC,
    "`": 0xC0,
}

#: 动作标识
ACTION_TOGGLE_TRANSLATION = "toggle_translation"
ACTION_SWITCH_MODEL = "switch_model"
ACTION_COPY_RESULT = "copy_result"
ACTION_TOGGLE_OVERLAY = "toggle_overlay"

ACTION_LABELS: dict[str, str] = {
    ACTION_TOGGLE_TRANSLATION: "开启 / 暂停实时翻译",
    ACTION_SWITCH_MODEL: "切换极速 / 高精度模型",
    ACTION_COPY_RESULT: "复制当前屏幕翻译结果",
    ACTION_TOGGLE_OVERLAY: "隐藏 / 显示翻译层",
}


class HotkeyError(Exception):
    """热键注册失败。"""


@dataclass
class HotkeyBinding:
    """一个热键绑定。"""

    action: str
    sequence: str
    #: 注册成功后分配的系统 ID
    hotkey_id: int = 0
    #: 是否注册成功
    registered: bool = False
    #: 失败原因
    error: str = ""
    #: 原组合被占用时自动改用的来源；配置文件保持不变。
    fallback_from: str = ""


def _fallback_sequence(action: str, sequence: str, desired: dict[str, str]) -> str:
    """Only replace the default model-switch chord when its backup is free.

    Never steal a chord explicitly assigned to another action by the user.
    Other custom shortcuts keep the existing visible conflict behavior.
    """
    if action != ACTION_SWITCH_MODEL:
        return ""
    try:
        if parse_sequence(sequence) != parse_sequence("Ctrl+Alt+M"):
            return ""
        candidate = parse_sequence("Ctrl+Alt+F1")
        for other_action, other_sequence in desired.items():
            if other_action == action:
                continue
            try:
                if parse_sequence(other_sequence) == candidate:
                    return ""
            except HotkeyError:
                continue
    except HotkeyError:
        return ""
    return "Ctrl+Alt+F1"


def parse_sequence(sequence: str) -> tuple[int, int]:
    """把 ``"Ctrl+Alt+T"`` 解析为 ``(修饰键位, 虚拟键码)``。

    :raises HotkeyError: 序列为空、缺少主键或键名无法识别。
    """
    if not sequence or not sequence.strip():
        raise HotkeyError("热键为空")

    text = sequence.strip()
    literal_plus = text == "+" or text.endswith("++")
    parts = [part.strip().lower() for part in text.split("+") if part.strip()]
    if literal_plus:
        parts.append("plus")
    if not parts:
        raise HotkeyError(f"无法解析热键：{sequence}")

    modifiers = 0
    main_key: str | None = None

    for part in parts:
        if part in _MODIFIERS:
            modifiers |= _MODIFIERS[part]
        else:
            if main_key is not None:
                raise HotkeyError(f"热键包含多个主键：{sequence}")
            main_key = part

    if main_key is None:
        raise HotkeyError(f"热键缺少主键：{sequence}")

    vk = _virtual_key_code(main_key)
    if vk is None:
        raise HotkeyError(f"无法识别的按键：{main_key}")

    return modifiers, vk


def _virtual_key_code(name: str) -> int | None:
    """把键名转换为虚拟键码。"""
    if name in _SPECIAL_KEYS:
        return _SPECIAL_KEYS[name]

    # 功能键 F1 - F24
    if name.startswith("f") and name[1:].isascii() and name[1:].isdigit():
        index = int(name[1:])
        if 1 <= index <= 24:
            return 0x70 + index - 1
        return None

    # 单个字母或数字：虚拟键码即其大写 ASCII 值
    if len(name) == 1 and name.isascii() and (name.isalpha() or name.isdigit()):
        return ord(name.upper())

    return None


def format_sequence(sequence: str) -> str:
    """把热键字符串规范化显示，例如 ``ctrl+alt+t`` → ``Ctrl+Alt+T``。"""
    try:
        parts = [part.strip() for part in sequence.split("+") if part.strip()]
    except AttributeError:
        return sequence

    pretty: list[str] = []
    for part in parts:
        lowered = part.lower()
        if lowered in {"ctrl", "control"}:
            pretty.append("Ctrl")
        elif lowered == "alt":
            pretty.append("Alt")
        elif lowered == "shift":
            pretty.append("Shift")
        elif lowered in {"win", "meta", "super"}:
            pretty.append("Win")
        else:
            pretty.append(part.upper() if len(part) == 1 else part.capitalize())
    return "+".join(pretty)


class _HotkeyThread(threading.Thread):
    """在独立线程中注册热键并泵送消息。

    Windows 的 ``RegisterHotKey`` 会把消息投递到**调用线程**的消息队列，
    因此注册与 ``GetMessageW`` 必须在同一线程执行。
    """

    def __init__(self, owner: HotkeyService) -> None:
        super().__init__(name="HotkeyMessageLoop", daemon=True)
        self._owner = owner
        self._ready = threading.Event()
        self._stop_event = threading.Event()
        self._thread_id: int | None = None
        self._bindings: list[HotkeyBinding] = []
        self._error: str = ""

    @property
    def thread_id(self) -> int | None:
        """消息循环线程的系统线程 ID。"""
        return self._thread_id

    @property
    def error(self) -> str:
        """注册阶段的错误信息。"""
        return self._error

    def run(self) -> None:
        """线程主体：注册热键后进入消息循环。"""
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()

        hotkey_id = 1
        desired = self._owner.desired_bindings()
        for action, sequence in desired.items():
            binding = HotkeyBinding(action=action, sequence=sequence)
            try:
                modifiers, vk = parse_sequence(sequence)
            except HotkeyError as exc:
                binding.error = str(exc)
                logger.warning("热键「%s」配置无效：%s", ACTION_LABELS.get(action, action), exc)
                self._bindings.append(binding)
                continue

            ok = user32.RegisterHotKey(None, hotkey_id, modifiers | _MOD_NOREPEAT, vk)
            if not ok:
                fallback = _fallback_sequence(action, sequence, desired)
                if fallback:
                    fallback_modifiers, fallback_vk = parse_sequence(fallback)
                    ok = user32.RegisterHotKey(
                        None, hotkey_id, fallback_modifiers | _MOD_NOREPEAT, fallback_vk
                    )
                    if ok:
                        binding.fallback_from = sequence
                        binding.sequence = fallback
                        logger.warning(
                            "热键 %s 被占用，已自动改用 %s",
                            format_sequence(sequence), format_sequence(fallback),
                        )
            if ok:
                binding.hotkey_id = hotkey_id
                binding.registered = True
                hotkey_id += 1
            else:
                # 注册失败最常见的原因是热键已被其他程序占用
                binding.error = "热键已被其他程序占用"
                logger.warning(
                    "热键注册失败（可能被占用）：%s → %s",
                    ACTION_LABELS.get(action, action),
                    format_sequence(sequence),
                )
            self._bindings.append(binding)

        self._ready.set()

        message = wintypes.MSG()
        while not self._stop_event.is_set():
            # 使用 PeekMessage 而非阻塞的 GetMessage，以便及时响应停止请求
            result = user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 1)
            if result:
                if message.message == _WM_HOTKEY:
                    self._dispatch(int(message.wParam))
                else:
                    user32.TranslateMessage(ctypes.byref(message))
                    user32.DispatchMessageW(ctypes.byref(message))
            else:
                # 队列空闲：短暂让出 CPU，避免忙等
                self._stop_event.wait(0.02)

        self._unregister_all()

    def _dispatch(self, hotkey_id: int) -> None:
        """把系统热键 ID 映射回动作并回调上层。"""
        for binding in self._bindings:
            if binding.hotkey_id == hotkey_id and binding.registered:
                try:
                    self._owner.trigger(binding.action)
                except Exception:  # noqa: BLE001 - 回调异常不得终止消息循环
                    logger.exception("热键回调执行失败：%s", binding.action)
                return

    def _unregister_all(self) -> None:
        """注销全部热键。"""
        user32 = ctypes.windll.user32
        for binding in self._bindings:
            if binding.registered:
                user32.UnregisterHotKey(None, binding.hotkey_id)
                binding.registered = False
        self._bindings = []

    def wait_ready(self, timeout: float = 3.0) -> bool:
        """等待注册阶段完成。"""
        return self._ready.wait(timeout)

    def bindings(self) -> list[HotkeyBinding]:
        """返回注册结果。"""
        return list(self._bindings)

    def stop(self) -> None:
        """请求停止消息循环。"""
        self._stop_event.set()


class HotkeyService:
    """全局热键服务。

    用法::

        service = HotkeyService(on_action=callback)
        service.start({ACTION_TOGGLE_TRANSLATION: "Ctrl+Alt+T", ...})
        ...
        service.stop()

    ``on_action`` 会在**热键线程**中被调用；若回调需要操作界面，
    必须自行切换到主线程（例如通过 Qt 信号）。
    """

    def __init__(self, on_action: Callable[[str], None] | None = None) -> None:
        self._on_action = on_action
        self._thread: _HotkeyThread | None = None
        self._bindings_config: dict[str, str] = {}
        self._enabled = True

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """热键是否已注册并在监听。"""
        return self._thread is not None and self._thread.is_alive()

    @property
    def enabled(self) -> bool:
        """热键功能是否被用户启用。"""
        return self._enabled

    def is_supported(self) -> bool:
        """当前平台是否支持全局热键。"""
        return _IS_WINDOWS

    def desired_bindings(self) -> dict[str, str]:
        """当前期望注册的热键表（供线程读取）。"""
        return dict(self._bindings_config)

    def bindings(self) -> list[HotkeyBinding]:
        """返回实际的注册结果，用于设置面板展示冲突提示。"""
        if self._thread is None:
            return []
        return self._thread.bindings()

    # ------------------------------------------------------------------
    # 启停
    # ------------------------------------------------------------------

    def start(self, bindings: dict[str, str] | None = None) -> bool:
        """注册热键并开始监听。

        :param bindings: ``{动作: 热键字符串}``；省略时沿用上次配置。
        :return: 是否成功启动（至少注册成功一个热键）。
        """
        if not _IS_WINDOWS:
            logger.info("当前平台不支持全局热键，功能已跳过")
            return False

        if bindings is not None:
            self._bindings_config = dict(bindings)

        if not self._bindings_config:
            logger.debug("未配置任何热键")
            return False

        self.stop()
        self._thread = _HotkeyThread(self)
        self._thread.start()

        if not self._thread.wait_ready(timeout=3.0):
            logger.warning("热键注册超时")
            self.stop()
            return False

        registered = [b for b in self._thread.bindings() if b.registered]
        if registered:
            summary = ", ".join(
                f"{ACTION_LABELS.get(b.action, b.action)}={format_sequence(b.sequence)}"
                for b in registered
            )
            logger.info("已注册 %s 个全局热键：%s", len(registered), summary)
        else:
            logger.warning("没有任何全局热键注册成功")
            self.stop()

        return bool(registered)

    def stop(self) -> None:
        """注销热键并结束监听线程。"""
        if self._thread is None:
            return
        self._thread.stop()
        self._thread.join(timeout=2.0)
        self._thread = None

    def restart(self, bindings: dict[str, str] | None = None) -> bool:
        """重新注册（用户修改热键后调用）。"""
        self.stop()
        return self.start(bindings)

    def set_enabled(self, enabled: bool) -> bool:
        """启用或禁用热键。

        :return: 变更后热键是否处于活动状态。
        """
        self._enabled = bool(enabled)
        if not self._enabled:
            self.stop()
            logger.info("全局热键已禁用")
            return False
        return self.start()

    # ------------------------------------------------------------------
    # 触发
    # ------------------------------------------------------------------

    def trigger(self, action: str) -> None:
        """触发一个热键动作（也被"编辑热键"对话框用于试触发）。"""
        if action not in ACTION_LABELS:
            logger.debug("收到未知热键动作：%s", action)
            return
        logger.debug("热键触发：%s", ACTION_LABELS[action])
        if self._on_action is not None:
            self._on_action(action)

    def set_callback(self, callback: Callable[[str], None]) -> None:
        """替换动作回调。"""
        self._on_action = callback


def default_bindings(config: object) -> dict[str, str]:
    """从配置节提取热键绑定表。

    :param config: :class:`~pingyitong.config.schema.HotkeyConfig`。
    """
    return {
        ACTION_TOGGLE_TRANSLATION: getattr(config, "toggle_translation", "Ctrl+Alt+T"),
        ACTION_SWITCH_MODEL: getattr(config, "switch_model", "Ctrl+Alt+M"),
        ACTION_COPY_RESULT: getattr(config, "copy_result", "Ctrl+Alt+C"),
        ACTION_TOGGLE_OVERLAY: getattr(config, "toggle_overlay", "Ctrl+Alt+H"),
    }
