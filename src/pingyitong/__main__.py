"""模块入口：``python -m pingyitong``。

支持的命令行参数：

``--portable``
    便携模式，把配置、日志、历史数据库、模型目录全部放在程序目录下的
    ``data/`` 中，便于放在 U 盘或移动硬盘里携带使用，不在系统盘留痕。

``--home <目录>``
    自定义应用数据目录，等价于设置 ``PINGYITONG_HOME`` 环境变量。

``--version``
    打印版本信息后退出。

``--help``
    打印帮助后退出。
"""

from __future__ import annotations

import os
import sys

_USAGE = """屏译通 —— 基于本地大模型的桌面实时原位翻译系统

用法：
    python -m pingyitong [选项]

选项：
    --portable        便携模式，数据保存在程序目录下的 data/ 中
    --home <目录>     指定应用数据目录
    --version         显示版本信息
    -h, --help        显示本帮助
    --self-check      自检：验证 OCR、模型权重、推理运行时是否就绪
"""


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    :return: 进程退出码。
    """
    argv = list(argv if argv is not None else sys.argv[1:])

    if "-h" in argv or "--help" in argv:
        print(_USAGE)
        return 0

    if "--version" in argv:
        from .version import version_string

        print(f"屏译通 {version_string()}")
        return 0

    # 处理 --home：转换为环境变量，由路径模块统一处理
    if "--home" in argv:
        index = argv.index("--home")
        if index + 1 < len(argv):
            os.environ["PINGYITONG_HOME"] = argv[index + 1]
            # 移除该参数及其取值，避免被 Qt 当作未知参数
            del argv[index : index + 2]

    if "--acceptance-output" in argv:
        index = argv.index("--acceptance-output")
        if index + 1 < len(argv) and os.environ.get("PINGYITONG_HOME"):
            os.environ["PINGYITONG_ACCEPTANCE_DIR"] = argv[index + 1]
            del argv[index:index + 2]

    portable = "--portable" in argv
    if portable:
        argv.remove("--portable")

    if "--self-check" in argv:
        from .selfcheck import run_self_check
        from .utils import paths

        paths.set_portable(portable)
        return run_self_check()

    # 提前校验 GUI 依赖，给出比 ImportError 堆栈更清晰的提示
    try:
        import PyQt6  # noqa: F401
    except ImportError:
        print(
            "错误：未安装 PyQt6，无法启动图形界面。\n"
            "请先执行：pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    from .ui.app import run

    # 便携标志必须**显式**传下去：run() 收到的参数列表里已经没有
    # --portable（上面移除过了），若指望 run() 从 argv 里重新认出它，
    # 只要命令行还带任何其他参数，便携模式就会静默失效，
    # 配置与数据全部写到系统盘。
    return run(([sys.argv[0]] + argv) if argv else None, portable=portable)


if __name__ == "__main__":
    sys.exit(main())
