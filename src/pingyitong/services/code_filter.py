"""命令行 / 代码片段识别（自研）。

**为什么要识别**：屏幕上（尤其终端、日志、聊天里的代码块）经常出现
以命令语法为主体的文本——``cd "F:/讯飞" && cat > /tmp/x <<EOF`` 这类。
把它们送进翻译模型，产出的"译文"基本是乱码（实测把一条 shell 命令
翻成了"宫廷所AD"式的内容）；而提示词本来就要求这部分**原样保留**。
既然正确结果就是"不翻"，最便宜也最可靠的做法是在送模型之前识别出来
直接跳过——这既消掉了乱码，也省下了这部分推理时间。

**为什么用启发式而不是模型**：每帧都要对几十个段落判断，任何额外
推理都直接进延迟；而命令语法（``&&``、管道、重定向、选项开关、
路径扩展名）的特征非常稳定，规则足以覆盖。

**宁可漏判不可误判**：判据偏保守——只有"命令语法特征明确"的段落才跳过。
普通句子即使夹带少量符号也会被送翻；漏判的代价只是这一次翻译质量差，
误判的代价是整段正文不被翻译，后者严重得多。
"""

from __future__ import annotations

import re
import unicodedata

#: 命令语法特征：逻辑连接、管道、重定向、命令替换
_COMMAND_SYNTAX = re.compile(
    r"(&&|\|\||\|(?!\|)|>>|<<|2>&1|\$\(|`[^`]+`)"
)

#: 常见命令动词开头的行（cd / cat / grep / git / python / pip …）
_COMMAND_VERBS = (
    r"(?:cd|ls|cat|head|tail|grep|awk|sed|echo|export|kill|"
    r"curl|wget|git|python3?|pip\d?|node|npm|npx|dotnet|java|cargo|make|"
    r"docker|kubectl|ssh|scp|rm|mv|cp|mkdir|chmod|chown|tar|zip|unzip|"
    r"taskkill|tasklist|netstat|powershell|cmd)"
)
_COMMAND_START = re.compile(
    r"^\s*(?:sudo\s+)?" + _COMMAND_VERBS + r"\b",
    re.IGNORECASE,
)
_COMMAND_IN_CHAIN = re.compile(
    r"(?:&&|\|\||\|)\s*(?:sudo\s+)?" + _COMMAND_VERBS + r"\b",
    re.IGNORECASE,
)
# "Make the output --compact" and "Export the file ..." are ordinary
# instructions even though their first word also names a shell command.
_NATURAL_SENTENCE_START = re.compile(
    r"^\s*(?:please\b|(?:make|export|kill|head|tail|echo)\s+"
    r"(?:the|a|an|your|our|this|that|these|those)\b)",
    re.IGNORECASE,
)

#: 长选项（--flag）与赋值形态（KEY=value）
_FLAG_LIKE = re.compile(r"(?:^|\s)--?[a-zA-Z][\w-]*|\b[A-Z_]{2,}=")

#: 符号占比超过该阈值视为代码。0.35 是折中：正常句子很少超过 0.15，
#: 而命令行普遍在 0.3 以上。
MAX_SYMBOL_RATIO = 0.35


def looks_like_code(text: str) -> bool:
    """判断一段文本是否为命令行 / 代码 / 路径为主体，不应翻译。

    :param text: 待判断文本。
    :return: ``True`` 表示应原样保留、不送翻译模型。
    """
    stripped = (text or "").strip()
    if len(stripped) < 8:
        return False

    if _NATURAL_SENTENCE_START.match(stripped):
        return False

    # A path / URL beside a pipe is also common in navigation labels and
    # ordinary prose. Require an actual command at a chain boundary instead.
    has_syntax = bool(_COMMAND_SYNTAX.search(stripped))
    has_start = bool(_COMMAND_START.match(stripped))
    has_chain_command = bool(_COMMAND_IN_CHAIN.search(stripped))
    has_flags = len(_FLAG_LIKE.findall(stripped)) >= 2

    if has_syntax and (has_start or has_chain_command):
        return True
    if has_start and has_flags:
        return True

    # 符号占比极高（代码、混淆文本）。中文按单字计入分母，
    # 否则中文句子的标点占比会被高估。
    symbols = words = 0
    attached_to_word = False
    for char in stripped:
        if char.isalnum() or char == "_":
            words += 1
            attached_to_word = True
        elif attached_to_word and unicodedata.category(char).startswith("M"):
            # Vowels and accents attached to a word are part of its spelling,
            # not code punctuation. Ignore them in both counts so repeating
            # accents cannot dilute the ratio of actual code symbols.
            continue
        else:
            attached_to_word = False
            if not char.isspace():
                symbols += 1
    total = symbols + words
    return bool(total and symbols / total > MAX_SYMBOL_RATIO)
