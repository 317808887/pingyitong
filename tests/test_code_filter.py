"""命令行 / 代码片段过滤测试。

**要防的两个真实缺陷**：

1. 把 shell 命令送进翻译模型，产出"宫廷所AD"式的乱码——
   正确行为是原样保留（提示词本来就要求技术 token 不翻）。
2. 反过来把正常句子误判成代码、跳过翻译——正文不被翻译比一次
   差翻译严重得多，因此判据必须偏保守。
"""

from __future__ import annotations

import unittest

from pingyitong.services.code_filter import looks_like_code


class TestCommandLines(unittest.TestCase):
    """应当跳过的：命令语法明确的文本。"""

    def test_shell_chain(self) -> None:
        text = '正在执行 cd "F:/讯飞/项目源码" && cat > /tmp/repro.py <<\'PYEOF\' import'
        self.assertTrue(looks_like_code(text))

    def test_taskkill_pipeline(self) -> None:
        text = "taskkill /F /PID 38176 2>&1 | head -1; taskkill /F /PID 8552 2>&1"
        self.assertTrue(looks_like_code(text))

    def test_git_command(self) -> None:
        self.assertTrue(looks_like_code("git commit -m \"fix: bug\" --author=x"))

    def test_python_invocation(self) -> None:
        text = "python scripts/test_languages.py --model 4b --output docs/out.json"
        self.assertTrue(looks_like_code(text))

    def test_curl_with_flags(self) -> None:
        text = "curl -H 'Content-Type: application/json' -d '{\"a\":1}' http://x.cn/api"
        self.assertTrue(looks_like_code(text))

    def test_symbol_soup(self) -> None:
        """符号占比极高的混淆文本应被跳过。"""
        self.assertTrue(looks_like_code("}]) {[}} {) ({ [[}]) .}) ]{) ({]] }]}) )]}"))


class TestNormalText(unittest.TestCase):
    """不应跳过的：正常句子（误判会让正文不被翻译）。"""

    def test_chinese_sentence(self) -> None:
        self.assertFalse(looks_like_code("屏幕上的译文会实时汇总显示在里面，按阅读顺序排列。"))

    def test_english_sentence(self) -> None:
        self.assertFalse(looks_like_code(
            "The transformer architecture revolutionized natural language processing."
        ))

    def test_sentence_mentioning_a_command(self) -> None:
        """句子里引用了一条命令，但主体是自然语言——必须仍送翻译。"""
        self.assertFalse(looks_like_code(
            "你可以在终端里运行 python run.py 来启动程序，然后按照提示操作。"
        ))

    def test_sentence_with_a_path(self) -> None:
        self.assertFalse(looks_like_code(
            "配置文件保存在 AppData 目录下，包含语言与模型设置。"
        ))

    def test_short_text_is_never_skipped(self) -> None:
        """过短的文本信息量不足，跳过得不偿失。"""
        self.assertFalse(looks_like_code("cd .."))
        self.assertFalse(looks_like_code("OK"))

    def test_empty(self) -> None:
        self.assertFalse(looks_like_code(""))
        self.assertFalse(looks_like_code("   "))


if __name__ == "__main__":
    unittest.main()
