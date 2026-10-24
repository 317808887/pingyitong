#!/usr/bin/env python
"""验证 pyproject.toml、requirements.txt、requirements-dev.txt 的依赖一致性。

存在的理由：pyproject.toml 的 [project.dependencies] 与 requirements.txt
曾分别维护，导致 `pip install -e .` 与 `pip install -r requirements.txt`
装出两套环境。prefetch 子包真实 import rapidfuzz、PyQt6、charset-normalizer、
trafilatura，若任一文件漏掉，安装后会缺依赖。

检查规则：
    1. requirements.txt 中列出的每个运行依赖（去掉版本限制后），
       必须在 pyproject.toml [project.dependencies] 中出现；
    2. pyproject.toml [project.dependencies] 中的每个依赖，
       必须在 requirements.txt 中出现；
    3. requirements-dev.txt 中的额外开发依赖（不含 -r requirements.txt
       所引入的），必须在 pyproject.toml [project.optional-dependencies].dev
       中出现，反之亦然。

用法::

    python scripts/verify_deps_sync.py

退出码 0 表示一致；1 表示存在差异，逐条打印。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"
REQUIREMENTS_DEV = ROOT / "requirements-dev.txt"

#: 版本说明分隔符，如 "PyQt6>=6.5,<7.0" -> "PyQt6"
_VERSION_SPLIT = re.compile(r"[<>=!~]")

#: requirements-dev.txt 中表示“引入基础依赖”的行
_DEV_BASE_INCLUDE = "-r requirements.txt"


def _parse_requirements(path: Path) -> dict[str, str]:
    """解析 requirements 风格文件，返回 {name: 原始行}。

    跳过空行、注释行、-r/--index-url 等选项行。
    """
    deps: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # 环境标记如 "pkg; python_version<'3.10'"，取分号前
        pkg = line.split(";")[0].strip()
        name = _VERSION_SPLIT.split(pkg, maxsplit=1)[0].strip()
        if name:
            deps[name.lower()] = line
    return deps


def _extract_toml_array(text: str, table_header: str, key: str) -> list[str]:
    """从 TOML 文本中提取指定表下某个键的数组项。

    仅做文本层面提取，避免依赖 toml 库（static job 无第三方依赖时仍可运行）。

    :param text: pyproject.toml 全文
    :param table_header: 表头，如 "[project]" 或 "[project.optional-dependencies]"
    :param key: 键名，如 "dependencies" 或 "dev"
    :return: 数组中的每一项原始字符串列表
    """
    lines = text.splitlines()
    items: list[str] = []
    in_table = False
    in_array = False
    brace_depth = 0

    for line in lines:
        stripped = line.strip()

        # 进入目标表
        if stripped == table_header:
            in_table = True
            continue

        # 离开当前表（遇到另一个 [xxx] 表头）
        if in_table and stripped.startswith("[") and stripped.endswith("]"):
            in_table = False
            in_array = False
            brace_depth = 0
            continue

        if not in_table:
            continue

        # 跳过注释和空行
        if not stripped or stripped.startswith("#"):
            continue

        if not in_array:
            # 寻找 key = [ ...
            if not stripped.startswith(key + " ="):
                continue
            rhs = stripped.split("=", 1)[1].strip()
            if not rhs.startswith("["):
                continue
            # 同一行内可能包含完整数组
            content = rhs[1:]
            brace_depth = content.count("[") - content.count("]")
            if "]" in content:
                # 数组在同一行内结束
                end = content.index("]")
                items.extend(_split_toml_array_items(content[:end]))
                in_array = False
                brace_depth = 0
            else:
                in_array = True
                # 取 [ 之后的部分
                items.extend(_split_toml_array_items(content))
        else:
            # 数组续行
            if "]" in stripped:
                end = stripped.index("]")
                items.extend(_split_toml_array_items(stripped[:end]))
                in_array = False
                brace_depth = 0
            else:
                items.extend(_split_toml_array_items(stripped))

    return items


def _split_toml_array_items(fragment: str) -> list[str]:
    """把 TOML 数组片段按逗号分割，去掉引号与尾随空白。"""
    result: list[str] = []
    for raw in fragment.split(","):
        item = raw.strip().strip('"').strip("'")
        if item:
            result.append(item)
    return result


def _parse_pyproject_dependencies(text: str) -> dict[str, str]:
    """从 pyproject.toml 文本中提取 [project] 下的 dependencies 列表。

    TOML 中 dependencies 是 [project] 表的一个键，而非独立节。
    """
    deps: dict[str, str] = {}
    for item in _extract_toml_array(text, "[project]", "dependencies"):
        pkg = item.split(";")[0].strip()
        name = _VERSION_SPLIT.split(pkg, maxsplit=1)[0].strip()
        if name:
            deps[name.lower()] = item
    return deps


def _parse_pyproject_dev_deps(text: str) -> dict[str, str]:
    """从 pyproject.toml 文本中提取 [project.optional-dependencies] 下的 dev 列表。"""
    deps: dict[str, str] = {}
    for item in _extract_toml_array(text, "[project.optional-dependencies]", "dev"):
        name = _VERSION_SPLIT.split(item, maxsplit=1)[0].strip().lower()
        if name:
            deps[name] = item
    return deps


def main() -> int:
    problems: list[str] = []

    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    req_deps = _parse_requirements(REQUIREMENTS)
    py_deps = _parse_pyproject_dependencies(pyproject_text)
    req_dev_deps = _parse_requirements(REQUIREMENTS_DEV)
    # 去掉 -r requirements.txt 引入的基础依赖，只保留 dev 额外依赖
    req_dev_extra = {k: v for k, v in req_dev_deps.items() if k != _DEV_BASE_INCLUDE}
    py_dev_deps = _parse_pyproject_dev_deps(pyproject_text)

    # 规则 1 & 2：requirements.txt <-> pyproject.toml [project.dependencies]
    for name in sorted(req_deps):
        if name not in py_deps:
            problems.append(
                f"requirements.txt 中有但 pyproject.toml [project.dependencies] 中缺少：{name} ({req_deps[name]})"
            )
    for name in sorted(py_deps):
        if name not in req_deps:
            problems.append(
                f"pyproject.toml [project.dependencies] 中有但 requirements.txt 中缺少：{name} ({py_deps[name]})"
            )

    # 规则 3：requirements-dev.txt 额外依赖 <-> pyproject.toml [project.optional-dependencies].dev
    for name in sorted(req_dev_extra):
        if name not in py_dev_deps:
            problems.append(
                f"requirements-dev.txt 中有但 pyproject.toml [project.optional-dependencies].dev 中缺少：{name} ({req_dev_extra[name]})"
            )
    for name in sorted(py_dev_deps):
        if name not in req_dev_extra:
            problems.append(
                f"pyproject.toml [project.optional-dependencies].dev 中有但 requirements-dev.txt 中缺少：{name} ({py_dev_deps[name]})"
            )

    if problems:
        print(f"依赖不同步：发现 {len(problems)} 处差异：\n")
        for problem in problems:
            print(f"  {problem}")
        print()
        return 1

    print(
        f"依赖同步通过：requirements.txt {len(req_deps)} 个运行依赖，"
        f"pyproject.toml [project.dependencies] {len(py_deps)} 个，"
        f"requirements-dev.txt 额外 {len(req_dev_extra)} 个开发依赖，"
        f"pyproject.toml dev {len(py_dev_deps)} 个，全部一致。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
