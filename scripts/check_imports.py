#!/usr/bin/env python
"""静态导入检查。

在**不导入依赖**的前提下校验全部源码的导入是否正确。之所以需要它：

- PyQt6 / onnxruntime / numpy 等依赖在部分环境（CI、评审机、仅有标准库的
  便携 Python）中并不存在，无法通过 ``import`` 验证；
- 包内相对导入的路径极易因重构而写错（``..services`` 写成 ``.services``），
  而这类错误只有在运行时才会暴露。

检查内容
    1. 所有相对导入的目标模块文件是否存在；
    2. ``from x import a, b`` 中的名字是否确实在 x 中定义或导出；
    3. 绝对导入的第三方包是否能被解析（可选，见 ``--check-third-party``）。

用法::

    python scripts/check_imports.py                 # 检查 src/ 与 tests/
    python scripts/check_imports.py --root src      # 只检查指定目录
    python scripts/check_imports.py --check-third-party

退出码：0 表示通过，1 表示发现问题。
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: 项目自身的顶层包名
PROJECT_PACKAGE = "pingyitong"

#: 依赖缺失时允许跳过校验的第三方包（它们可能未安装在当前环境）
OPTIONAL_THIRD_PARTY = {
    "PyQt6",
    "mss",
    "cv2",
    "numpy",
    "rapidocr_onnxruntime",
    "rapidocr",
    "llama_cpp",
    "psutil",
    "pynvml",
}


@dataclass
class ModuleInfo:
    """一个 Python 模块的静态信息。"""

    #: 模块点分名，例如 ``pingyitong.services.glossary_service``
    name: str
    #: 源文件路径
    path: Path
    #: 模块顶层定义的名字（类、函数、赋值目标）
    defined: set[str] = field(default_factory=set)
    #: 通过 ``from ... import`` 引入的名字（视为可再次导出）
    imported: set[str] = field(default_factory=set)
    #: 自身是否为包（目录 + ``__init__.py``）
    is_package: bool = False

    @property
    def exports(self) -> set[str]:
        """该模块对外可见的名字。"""
        return self.defined | self.imported


@dataclass
class Problem:
    """一处导入问题。"""

    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        try:
            relative = self.path.relative_to(Path.cwd())
        except ValueError:
            relative = self.path
        return f"{relative}:{self.line}: {self.message}"


def collect_modules(root: Path, package_name: str | None = None) -> dict[str, ModuleInfo]:
    """扫描目录，收集全部模块及其导出名。

    :param root: 源码根目录（其中包含顶层包目录，或本身就是包目录）。
    :param package_name: 顶层包名；省略时按目录结构推断。
    """
    modules: dict[str, ModuleInfo] = {}

    for path in sorted(root.rglob("*.py")):
        # 跳过构建产物与缓存
        if any(part in {"__pycache__", "build", "dist", ".venv", "venv"} for part in path.parts):
            continue

        module_name = _module_name_for(path, root, package_name)
        if module_name is None:
            continue

        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError) as exc:
            # 语法错误由 py_compile 负责报告，这里只跳过
            print(f"跳过（无法解析）：{path}：{exc}", file=sys.stderr)
            continue

        info = ModuleInfo(
            name=module_name,
            path=path,
            is_package=path.name == "__init__.py",
        )
        _collect_names(tree, info)
        modules[module_name] = info

    return modules


def _module_name_for(path: Path, root: Path, package_name: str | None) -> str | None:
    """把文件路径转换为模块点分名。"""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None

    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    if not parts:
        return package_name

    if package_name is None:
        return ".".join(parts)

    # root 本身是包目录时，加上包名前缀
    if root.name == package_name:
        return ".".join([package_name] + parts)
    return ".".join(parts)


def _collect_names(tree: ast.Module, info: ModuleInfo) -> None:
    """收集模块顶层的定义名与导入名。"""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            info.defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                info.defined |= _target_names(target)
        elif isinstance(node, ast.AnnAssign):
            info.defined |= _target_names(node.target)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                info.imported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    info.imported.add(alias.asname or alias.name)


def _target_names(node: ast.expr) -> set[str]:
    """提取赋值目标的名称。"""
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in node.elts:
            names |= _target_names(element)
        return names
    return set()


def check_module(
    info: ModuleInfo,
    modules: dict[str, ModuleInfo],
    package_root: str,
) -> list[Problem]:
    """检查单个模块的导入。"""
    problems: list[Problem] = []

    try:
        tree = ast.parse(info.path.read_text(encoding="utf-8"), filename=str(info.path))
    except (OSError, SyntaxError):
        return problems

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            continue  # 绝对导入由 _check_absolute 处理

        target_name = _resolve_relative(info, node, package_root)
        if target_name is None:
            problems.append(
                Problem(info.path, node.lineno, "相对导入超出了包的范围")
            )
            continue

        target = modules.get(target_name)
        if target is None:
            problems.append(
                Problem(
                    info.path,
                    node.lineno,
                    f"导入的模块不存在：{target_name}",
                )
            )
            continue

        for alias in node.names:
            if alias.name == "*":
                continue
            if alias.name in target.exports:
                continue
            # 也允许导入子模块（例如 ``from . import widgets``）
            if f"{target_name}.{alias.name}" in modules:
                continue
            problems.append(
                Problem(
                    info.path,
                    node.lineno,
                    f"模块 {target_name} 中不存在名字 {alias.name!r}",
                )
            )

    return problems


def _resolve_relative(
    info: ModuleInfo,
    node: ast.ImportFrom,
    package_root: str,
) -> str | None:
    """把相对导入解析为绝对模块名。

    ``level`` 表示向上回溯的层数：``level=1`` 指当前包，
    ``level=2`` 指上一级包，依此类推。
    """
    # 当前模块所属的包：包自身即为其名，普通模块需去掉最后一段
    package_parts = (
        info.name.split(".") if info.is_package else info.name.split(".")[:-1]
    )

    # 回溯 level-1 层
    up = node.level - 1
    if up > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - up]

    if node.module:
        base_parts = base_parts + node.module.split(".")

    resolved = ".".join(base_parts)
    return resolved or package_root


def check_third_party(modules: dict[str, ModuleInfo], available: set[str]) -> list[Problem]:
    """检查绝对导入的第三方顶层包是否可用。"""
    problems: list[Problem] = []

    for info in modules.values():
        try:
            tree = ast.parse(info.path.read_text(encoding="utf-8"), filename=str(info.path))
        except (OSError, SyntaxError):
            continue

        for node in ast.walk(tree):
            names: list[tuple[str, int]] = []
            if isinstance(node, ast.Import):
                names = [(alias.name.split(".")[0], node.lineno) for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [(node.module.split(".")[0], node.lineno)]

            for name, lineno in names:
                if name in PROJECT_PACKAGE or name in available:
                    continue
                if name in OPTIONAL_THIRD_PARTY:
                    problems.append(
                        Problem(
                            info.path,
                            lineno,
                            f"可选依赖 {name} 在当前环境中不可用"
                            f"（运行时已做降级处理，此为提示）",
                        )
                    )
                else:
                    problems.append(
                        Problem(info.path, lineno, f"未知的第三方依赖：{name}")
                    )

    return problems


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="静态导入检查")
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="要检查的根目录，可重复指定（默认 src 与 tests）",
    )
    parser.add_argument(
        "--check-third-party",
        action="store_true",
        help="同时检查第三方依赖是否可用",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    roots = [Path(item) for item in args.root] if args.root else [
        project_root / "src",
        project_root / "tests",
    ]
    roots = [item if item.is_absolute() else project_root / item for item in roots]

    modules: dict[str, ModuleInfo] = {}
    for root in roots:
        if not root.exists():
            print(f"跳过不存在的目录：{root}", file=sys.stderr)
            continue
        modules.update(collect_modules(root, PROJECT_PACKAGE if "src" in str(root) else None))

    if not modules:
        print("未找到任何 Python 模块", file=sys.stderr)
        return 1

    problems: list[Problem] = []
    for info in modules.values():
        problems.extend(check_module(info, modules, PROJECT_PACKAGE))

    if args.check_third_party:
        problems.extend(check_third_party(modules, set(sys.modules)))

    if problems:
        print(f"发现 {len(problems)} 处导入问题：\n")
        for problem in problems:
            print(f"  {problem}")
        print()
        return 1

    print(f"导入检查通过：共检查 {len(modules)} 个模块。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
