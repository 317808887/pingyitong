#!/usr/bin/env python
"""屏译通启动脚本。

用法::

    python run.py                # 正常启动
    python run.py --portable     # 便携模式
    python run.py --version      # 查看版本

该脚本会把 ``src`` 目录加入模块搜索路径，因此无需安装即可直接运行，
便于开发与调试。正式分发的安装包由 ``build/build_exe.py`` 生成。
"""

from __future__ import annotations

import sys
from pathlib import Path

#: 把 src 加入搜索路径，使源码目录下可直接运行
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _check_python_version() -> None:
    """检查 Python 版本。

    项目使用 ``X | Y`` 联合类型语法与 ``dataclass`` 特性，
    最低需要 Python 3.10。
    """
    # UP036 豁免：requires-python 只约束 pip 安装场景，本脚本支持免安装直接
    # 运行，必须在进入 src 包之前对低版本 Python 给出友好报错。
    if sys.version_info < (3, 10):  # noqa: UP036
        print(
            f"错误：需要 Python 3.10 或更高版本，当前为 "
            f"{sys.version_info.major}.{sys.version_info.minor}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def main() -> int:
    """启动程序。"""
    _check_python_version()

    from pingyitong.__main__ import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
