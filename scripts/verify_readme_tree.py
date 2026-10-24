#!/usr/bin/env python
"""校验 README 的「项目结构」树与仓库实际内容一致。

存在的理由：README 的结构树曾漏掉 docs/ 下 11 个已跟踪文件、database/、
CHANGELOG.md 与三个根目录过程报告，评审按图索骥会以为文档已过期。
本脚本把这种漂移变成 CI 红灯，而不是等评审发现。

判定规则：
    - 树中列出的每个路径都必须存在（glob 写法除外，如 ``*.json``）；
    - 顶层已跟踪目录若未在树中出现，报缺失。

用法::

    python scripts/verify_readme_tree.py

退出码 0 表示一致；1 表示存在漂移，逐条打印差异。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

#: 结构树章节的起止标题
_TREE_START = "## 项目结构"
_TREE_END = "## 技术架构"

#: 树根目录名（第一层节点，不属于仓库相对路径）
_TREE_ROOT_NAME = "项目源码"


def _tree_block() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index(_TREE_START)
    end = text.index(_TREE_END)
    return text[start:end]


def _parse_tree(block: str) -> list[str]:
    """把树形文本解析为相对路径列表。"""
    paths: list[str] = []
    stack: list[str] = []
    pattern = re.compile(r"^((?:[│ ]   )*)[├└]── ([^\s#]+)")
    for line in block.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        depth = len(match.group(1)) // 4
        name = match.group(2).rstrip("/")
        stack = stack[:depth]
        stack.append(name)
        # 第一层是树根（项目源码/），跳过
        if stack[0] == _TREE_ROOT_NAME:
            stack = stack[1:]
            if not stack:
                continue
        paths.append("/".join(stack))
    return paths


def _tracked_top_level() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
    )
    # -z 让 git 用 NUL 分隔且不给非 ASCII 路径加引号；
    # 不加 -z 时输出形如 "docs/xxx"，剥不掉那双引号会造出幽灵条目。
    tops: set[str] = set()
    for line in result.stdout.split("\0"):
        if not line:
            continue
        parts = line.replace("\\", "/").split("/")
        tops.add(parts[0] if len(parts) == 1 else parts[0] + "/")
    return tops


def main() -> int:
    block = _tree_block()
    listed = _parse_tree(block)

    # 规则一：树中列出的路径必须存在（跳过 glob 写法）
    missing = [
        path
        for path in listed
        if "*" not in path and not (ROOT / path).exists()
    ]

    # 规则二：顶层已跟踪目录/文件必须在树中出现
    listed_tops = {path.split("/")[0].rstrip("/") for path in listed}
    listed_tops |= {p.rstrip("/") for p in listed if "/" not in p}
    omitted = sorted(
        top
        for top in _tracked_top_level()
        if top.rstrip("/") not in listed_tops
    )

    if not missing and not omitted:
        print(f"结构树一致：{len(listed)} 个条目全部存在，顶层无遗漏。")
        return 0

    for path in missing:
        print(f"树中存在但仓库没有: {path}")
    for top in omitted:
        print(f"已跟踪但树中未列出: {top}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
