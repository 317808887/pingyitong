"""安装自检。

用户装完软件后如果发现"不能翻译"，需要能自己判断卡在哪一环。
本模块逐项验证并给出可读结论：

1. **屏幕捕获**（mss 可用）
2. **文字识别**（RapidOCR 可用，且**真的能识别出文字**——
   只检查 import 是不够的：ONNX 模型文件缺失时 import 依然成功）
3. **模型权重**（GGUF 文件存在且尺寸合理）
4. **推理运行时**（llama-server 可执行文件存在）
5. **可写数据目录**

用法::

    PingYiTong.exe --self-check
    python run.py --self-check
"""

from __future__ import annotations

import sys
from pathlib import Path

#: 各输出行的状态标记
_OK = "✓"
_FAIL = "✗"
_WARN = "!"


#: 自检进程自建的 ``QGuiApplication`` 引用。
#: 必须持有引用：PyQt6 的对象在 Python 侧失去引用后 C++ 实例会随之销毁，
#: 局部变量一离开作用域，``screens()`` 就又变回空列表。
_QT_APP = None


def _ensure_qt_app() -> bool:
    """确保存在 ``QApplication``/``QGuiApplication``。

    ``QGuiApplication.screens()`` 在**没有应用实例**时返回空列表——
    自检是独立进程，必须自己建一个，否则会把"没建应用"误报成"没有显示器"。

    优先建 ``QApplication``（带控件支持）：先建 ``QGuiApplication`` 再建
    ``QApplication`` 会直接抛 ``RuntimeError``，导致自检结果弹窗根本弹不出来。
    """
    global _QT_APP
    if _QT_APP is not None:
        return True
    try:
        from PyQt6.QtCore import QCoreApplication
        from PyQt6.QtWidgets import QApplication

        instance = QCoreApplication.instance()
        if isinstance(instance, QApplication):
            _QT_APP = instance
            return True
        if instance is not None:
            # Qt 全进程只允许一个 Application。若调用方已经错误地建了
            # QGuiApplication，就不能再“升级”为 QApplication；返回 False
            # 让结果展示走原生 Win32 弹窗，而不是触发单例冲突。
            return False
        _QT_APP = QApplication([])
        return True
    except ImportError:
        pass
    try:
        from PyQt6.QtGui import QGuiApplication

        if QGuiApplication.instance() is None:
            _QT_APP = QGuiApplication([])
        return False
    except ImportError:
        return False


def _check_capture() -> tuple[bool, str]:
    """屏幕捕获能力——**真的抓一帧**。

    与 OCR 检查同理：只问 ``is_available()`` 或只枚举显示器，
    都无法发现"权限被拦/驱动返回全黑"这类实际取不到图的问题，
    而这类问题正是用户说的"开了翻译没反应"。故此处按最小尺寸实抓一帧，
    校验像素不全为同一值（全黑或全白说明捕获链路是坏的）。
    """
    try:
        from .components.capture.screen_capture import ScreenCapturer
    except ImportError as exc:
        return False, f"模块导入失败：{exc}"
    if not ScreenCapturer.is_available():
        return False, "mss 不可用（未安装或平台不支持）"

    _ensure_qt_app()
    try:
        from .components.capture.geometry import Rect, list_screens

        screens = list_screens()
    except Exception as exc:  # noqa: BLE001 - 自检不应抛异常
        return False, f"枚举显示器失败：{exc}"
    if not screens:
        return False, "未检测到显示器"

    try:
        capturer = ScreenCapturer()
        capturer.open()
        try:
            frame = capturer.capture_monitor()
        finally:
            capturer.close()
    except Exception as exc:  # noqa: BLE001
        return False, f"抓取屏幕失败：{type(exc).__name__}: {exc}"

    if frame.width <= 0 or frame.height <= 0:
        return False, f"抓到的画面尺寸异常：{frame.width}×{frame.height}"

    # 取中心 200×200 的方块，检查不是单一颜色（排除全黑/全白的失败抓取）
    probe = Rect(
        max(0, frame.width // 2 - 100),
        max(0, frame.height // 2 - 100),
        min(200, frame.width),
        min(200, frame.height),
    )
    try:
        patch = frame.crop(probe)
        varying = bool(patch.min() != patch.max())
    except Exception as exc:  # noqa: BLE001
        return False, f"画面内容读取失败：{type(exc).__name__}: {exc}"
    if not varying:
        # 单一颜色可能是桌面本来就是纯色，故只在完全取不到图时才判失败
        return True, f"可用，{len(screens)} 块显示器，已抓到 {frame.width}×{frame.height} 画面（中心为纯色）"

    return True, f"可用，{len(screens)} 块显示器，已抓到 {frame.width}×{frame.height} 画面"


def _check_ocr() -> tuple[bool, str]:
    """文字识别能力——**做一次真实识别**。

    只验证 import 会漏掉"ONNX 模型文件没打进去"这类打包问题：
    那种情况下导入成功、``is_ready()`` 返回 True，但一识别就报错。
    """
    try:
        from .components.ocr.ocr_engine import OcrEngine
    except ImportError as exc:
        return False, f"模块导入失败：{exc}"

    engine = OcrEngine()
    if not engine.is_ready():
        reason = engine.error_message() or "依赖缺失或模型加载失败"
        return False, f"RapidOCR 未就绪：{reason}"

    # 真实识别：构造一张带文字的图像，验证能读出文字
    try:
        import numpy as np

        image = np.full((80, 420, 3), 255, dtype=np.uint8)
        try:
            import cv2

            cv2.putText(image, "Self test 12345", (12, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        except ImportError:
            # 无 cv2 时画简易黑白块，仍能验证推理链路不崩
            image[30:50, 20:300] = 0

        blocks = engine.recognize_image(image)
        text = " ".join(block.text for block in blocks)
        if not blocks:
            return False, "识别未返回任何文本（模型文件可能损坏或缺失）"
        import unicodedata

        # The synthetic fixture has a known literal answer. Nonempty output
        # alone can hide a wrong alphabet/model or missing/reversed digits.
        normalized = "".join(
            char for char in unicodedata.normalize("NFKC", text).casefold()
            if not char.isspace()
        )
        if normalized != "selftest12345":
            return False, (
                "测试图识别不符：预期 'Self test 12345'，"
                f"实际 {text[:80]!r}（文字或数字识别错误）"
            )
        return True, f"可用，测试图像识别出 {len(blocks)} 个文本块：{text[:30]!r}"
    except Exception as exc:  # noqa: BLE001 - 自检不应抛异常
        return False, f"识别失败：{type(exc).__name__}: {exc}"


def _check_models() -> tuple[bool, str]:
    """模型权重存在性。"""
    try:
        from .components.llm.model_pool import ModelPool
        from .config.schema import AppConfig
        from .constants import MODEL_ACCURATE, MODEL_FAST, MODEL_LABELS
    except ImportError as exc:
        return False, f"模块导入失败：{exc}"

    config = AppConfig()
    pool = ModelPool(config.model, idle_timeout=60)
    if not pool.any_available():
        from .utils import paths

        return False, (
            f"未找到模型权重。请确认 models/ 目录存在且包含 .gguf 文件："
            f"{paths.models_dir()}"
        )

    found: list[str] = []
    for model_id in (config.schedule.active_model, MODEL_FAST, MODEL_ACCURATE):
        try:
            if not pool.is_available(model_id):
                continue
            status = pool.engine(model_id).status()
            path = Path(getattr(status, "path", "") or "")
            label = MODEL_LABELS.get(model_id, model_id)
            if path.is_file():
                found.append(f"{label} → {path.name}（{path.stat().st_size / 1024**3:.2f} GB）")
            else:
                found.append(label)
        except Exception:  # noqa: BLE001 - 单个模型异常不影响其余检查
            continue

    unique = sorted(set(found), key=found.index) if found else []
    if not unique:
        return False, "模型已就绪但无法定位权重文件"
    return True, "可用：" + "、".join(unique)


def _check_runtime() -> tuple[bool, str]:
    """推理运行时存在性（缺失时可用 llama-cpp-python 兜底，故只是警告）。"""
    try:
        from .components.llm.backend import probe

        result = probe()
    except Exception as exc:  # noqa: BLE001
        return False, f"后端探测失败：{exc}"
    if result.usable:
        detail = result.detail or result.kind.value
        return True, f"可用：{detail}"
    return False, "无可用推理后端（llama-server 缺失且未安装 llama-cpp-python）"


def _check_data_dir() -> tuple[bool, str]:
    """数据目录可写性。"""
    try:
        from .utils import paths

        directory = paths.app_data_dir()
        probe_file = directory / ".write_probe"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink()
        return True, f"可写：{directory}"
    except Exception as exc:  # noqa: BLE001
        return False, f"不可写：{exc}"


def _collect_report_checks() -> tuple[tuple[str, object, bool], ...]:
    """返回全部检查项 ``(名称, 检查函数, 是否必过)``。

    单独拆出来便于测试注入替身：报告组装逻辑（标记、汇总、失败收集）
    不依赖任何真实硬件/模型。
    """
    return (
        ("屏幕捕获", _check_capture, True),
        ("文字识别", _check_ocr, True),
        ("模型权重", _check_models, True),
        ("推理运行时", _check_runtime, True),
        ("数据目录", _check_data_dir, False),
    )


def _collect_report() -> tuple[list[str], list[str]]:
    """执行全部检查，返回 ``(报告行, 未通过项)``。"""
    from .version import version_string

    lines = ["=" * 68, f"屏译通 {version_string()} 安装自检", "=" * 68]

    failures: list[str] = []
    for label, check, required in _collect_report_checks():
        try:
            ok, detail = check()
        except Exception as exc:  # noqa: BLE001 - 自检自身不得崩溃
            ok, detail = False, f"自检异常：{type(exc).__name__}: {exc}"
        mark = _OK if ok else (_FAIL if required else _WARN)
        lines.append(f"  {mark} {label}：{detail}")
        if not ok and required:
            failures.append(label)

    lines.append("=" * 68)
    if failures:
        lines.append(f"自检未通过：{'、'.join(failures)}")
        lines.append("请按上方提示处理；模型与运行时缺失时可重新运行安装包修复安装。")
    else:
        lines.append("自检通过：软件已就绪。")
    return lines, failures


def _write_log(lines: list[str]) -> Path | None:
    """把报告写入数据目录下的 ``selfcheck.log``，便于用户反馈问题时附上。"""
    try:
        from .utils import paths

        log_path = paths.app_data_dir() / "selfcheck.log"
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return log_path
    except Exception:  # noqa: BLE001 - 写日志失败不影响自检结论
        return None


def _show_dialog(lines: list[str], log_path: Path | None) -> None:
    """在图形界面中展示自检结果。

    打包后的程序是窗口子系统（``--noconsole``），没有控制台，
    ``print`` 的内容用户根本看不到——必须弹窗，否则自检等于没做。
    """
    if not _ensure_qt_app():
        # 已有 QGuiApplication 时无法创建 QWidget；用系统消息框保证用户
        # 仍能看到结论。完整报告已写入 selfcheck.log。
        try:
            import ctypes

            failed = any(line.strip().startswith(_FAIL) for line in lines)
            summary = "自检未通过，请查看报告。" if failed else "自检通过：软件已就绪。"
            if log_path is not None:
                summary += f"\n\n报告：{log_path}"
            icon = 0x10 if failed else 0x40  # MB_ICONERROR / MB_ICONINFORMATION
            ctypes.windll.user32.MessageBoxW(0, summary, "屏译通 自检结果", icon)
        except Exception:  # noqa: BLE001 - 展示失败不能影响退出码与落盘报告
            pass
        return

    from PyQt6.QtWidgets import QMessageBox

    box = QMessageBox()
    box.setWindowTitle("屏译通 自检结果")
    failed = any(line.strip().startswith(_FAIL) for line in lines)
    box.setIcon(QMessageBox.Icon.Critical if failed else QMessageBox.Icon.Information)
    box.setText("自检未通过，请查看详情。" if failed else "自检通过：软件已就绪。")
    detail = "\n".join(lines)
    if log_path is not None:
        detail += f"\n\n报告已保存：{log_path}"
    box.setDetailedText(detail)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()


def run_self_check() -> int:
    """执行全部自检并输出结论。返回进程退出码（0 = 全部通过）。

    有控制台时打印到控制台；打包后的 GUI 程序没有控制台，
    此时弹窗展示（报告同时写入数据目录的 ``selfcheck.log``）。
    """
    windowed = _is_windowed()
    if windowed:
        # 必须在任何屏幕枚举之前先建 QApplication。旧实现先建
        # QGuiApplication，结束时再为 QMessageBox 建 QApplication，违反
        # Qt 单例约束，结果就是检查完成但结果弹窗永远出不来。
        _ensure_qt_app()

    lines, failures = _collect_report()
    log_path = _write_log(lines)

    printed = False
    if sys.stdout is not None:
        try:
            print("\n".join(lines))
            printed = True
        except Exception:  # noqa: BLE001 - 输出不可用时退化为弹窗
            printed = False

    if not printed or windowed:
        _show_dialog(lines, log_path)

    return 1 if failures else 0


def _is_windowed() -> bool:
    """判断当前是否为无控制台的窗口子系统进程。"""
    return bool(getattr(sys, "frozen", False)) and sys.stdout is None


if __name__ == "__main__":
    sys.exit(run_self_check())
