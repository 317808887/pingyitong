"""分层依赖方向测试。

《项目全栈开发文档》把系统分为五层：UI 交互层 → 业务功能层 →
核心调度层 → 基础组件层 → 硬件层。依赖只能**自上而下**：

    ui/  →  services/  →  core/  →  components/

**实际发生过一次违反**：翻译 Prompt 需要语言名称表，而语言表当时放在
``services/languages.py``，于是 ``components/llm/prompt.py`` 反向依赖了
业务层，形成 ``components → services → components`` 的循环导入，
表现为"单独导入 components 就报 ImportError: cannot import name
'TranslationCache' from partially initialized module"。

修法不是把循环绕开（延迟导入之类），而是让语言表回到它该在的位置：
它是**数据**，与 ``constants.py`` 同级放在包根，各层都可以引用。

这个测试守住依赖方向，避免同类问题重现。
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

#: 源码根目录
_SRC = Path(__file__).resolve().parents[1] / "src" / "pingyitong"

#: 各层允许依赖的下层。键为层目录名，值为允许出现的层前缀。
_ALLOWED: dict[str, tuple[str, ...]] = {
    # 基础组件层不得依赖上层
    "components": ("components", "utils", "languages", "language_test_results", "constants"),
    # 核心调度层可以用组件层
    "core": ("core", "components", "utils", "languages", "constants"),
    # 业务功能层可以用核心层与组件层
    "services": (
        "services",
        "core",
        "components",
        "utils",
        "languages",
        "language_test_results",
        "constants",
    ),
}


def _iter_imports(path: Path) -> list[tuple[int, str]]:
    """解析文件，返回 ``(行号, 被导入模块的绝对名)`` 列表。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith("pingyitong"):
                    found.append((node.lineno, node.module))
                continue
            # 把相对导入换算成包内绝对名
            depth = node.level
            parts = path.relative_to(_SRC).parts[:-1]
            base = parts[: len(parts) - (depth - 1)] if depth > 1 else parts
            module = ".".join(base + ((node.module,) if node.module else ()))
            found.append((node.lineno, f"pingyitong.{module}" if module else "pingyitong"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("pingyitong"):
                    found.append((node.lineno, alias.name))

    return found


class TestLayerDependencies(unittest.TestCase):
    """依赖方向只能自上而下。"""

    def test_layers_do_not_import_upwards(self) -> None:
        """components/ 与 core/ 不得导入 services/ 或 ui/。"""
        violations: list[str] = []

        for layer, allowed in _ALLOWED.items():
            layer_dir = _SRC / layer
            if not layer_dir.is_dir():
                continue

            for path in layer_dir.rglob("*.py"):
                for lineno, module in _iter_imports(path):
                    parts = module.split(".")
                    # pingyitong.<层>.<...>
                    target = parts[1] if len(parts) > 1 else ""
                    if target in ("ui", "services") and target not in allowed:
                        rel = path.relative_to(_SRC.parent.parent)
                        violations.append(
                            f"{rel}:{lineno} 导入了 {module}（{layer} 层不得依赖 {target} 层）"
                        )

        self.assertEqual(violations, [], "发现违反分层方向的导入：\n" + "\n".join(violations))

    def test_language_table_is_importable_without_services(self) -> None:
        """语言表必须能独立导入，不得牵出业务层。

        这条断言直接对应那次循环导入：只要语言表还需要 ``services`` 包
        被完整初始化，就会把 ``components`` 拉进循环。
        """
        import subprocess
        import sys

        script = (
            rf"import sys; sys.path.insert(0, r'{_SRC.parent}');"
            "from pingyitong.languages import LANGUAGES, verification_status;"
            "print(len(LANGUAGES), verification_status('zh'))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, f"导入语言表失败：{proc.stderr}")
        count, status = proc.stdout.split()
        self.assertGreater(count and int(count), 0)
        self.assertIn(status, {"verified", "likely", "unsupported"})

    def test_prompt_module_imports_standalone(self) -> None:
        """翻译 Prompt 模块必须能独立导入（这是发生过循环导入的地方）。"""
        import subprocess
        import sys

        script = (
            rf"import sys; sys.path.insert(0, r'{_SRC.parent}');"
            "from pingyitong.components.llm.prompt import build_system_prompt;"
            "print(build_system_prompt('en', 'ja')[:40])"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, f"导入 Prompt 模块失败：{proc.stderr}")


if __name__ == "__main__":
    unittest.main()
