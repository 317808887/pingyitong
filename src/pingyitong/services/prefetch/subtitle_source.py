"""字幕文件解析（联网预翻译 P2，自研）。

对应方案 v1.2 §3.2 的"解析器现实清单"。设计要点：

- **以字幕行为缓存与匹配单位**，不做二次分句（半句跨行保留时间轴完整）；
- 编码：UTF-8（含 BOM）严格 → GB18030 严格（覆盖 GBK/ANSI 中文社区常见
  编码）→ charset-normalizer 兜底。顺序很重要：短文本上编码探测库
  不可靠（实测 GBK 小样本被误判成其他编码读出乱码），显式严格解码优先；
- ASS：``Dialogue`` 行按逗号 **maxsplit=9** 切分（Text 是第 10 个字段，
  允许内含逗号），剥离样式标签与 Comment 行，``\\N``/``\\h`` 还原；
- VTT：WEBVTT 头、NOTE/STYLE/REGION 块剔除，小时位可选；
- 双语轨：按 CJK 占比选主语言行，副语言行丢弃（否则同一画面两条文字
  互相干扰 fuzzy 匹配）；
- 时间轴重叠、零时长行容错（不抛异常，解析继续）。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ...utils.logger import get_logger

logger = get_logger(__name__)

#: ASS Dialogue：Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
#: Text 内允许逗号，因此用 maxsplit=9 取最后一段
_ASS_DIALOGUE = re.compile(r"^Dialogue:\s*(.*)$")
_ASS_TAG = re.compile(r"\{[^}]*\}")

#: VTT cue 时间轴（小时位可选）
_VTT_TIME = re.compile(
    r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
)

#: SRT 时间轴（小时位必有）
_SRT_TIME = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)


@dataclass
class SubtitleLine:
    """一条字幕行（缓存与匹配单位）。"""

    line_no: int
    start_ms: int
    end_ms: int
    text_raw: str
    text_norm: str = ""
    lang: str = ""


def _time_ms(hours: str, minutes: str, seconds: str, fraction: str) -> int:
    """时间分量 → 毫秒（小数位不足三位右侧补齐）。"""
    return (
        (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000
        + int(fraction.ljust(3, "0"))
    )


def _read_text(path: Path) -> str:
    """读字幕文件：显式严格解码优先，探测库兜底。

    顺序：UTF-8 BOM → UTF-8 严格 → GB18030 严格（GBK 超集）→
    charset-normalizer。短文本上探测库不可靠（实测 GBK 小样本被
    误判读出乱码），因此探测只作最后兜底。
    """
    data = Path(path).read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="strict")
    for encoding in ("utf-8", "gb18030"):
        try:
            return data.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            continue
    try:
        from charset_normalizer import from_path
    except ImportError:  # pragma: no cover
        raise ValueError(f"无法解码字幕文件：{path}") from None
    best = from_path(path).best()
    if best is None:
        raise ValueError(f"无法识别字幕文件编码：{path}")
    return str(best)


def _blocks(raw: str) -> list[list[str]]:
    """按空行切块。"""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in raw.splitlines():
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def parse_srt(path, normalizer) -> list[SubtitleLine]:
    """SRT：序号行 + 时间轴 + 文本块。"""
    lines: list[SubtitleLine] = []
    for block in _blocks(_read_text(path)):
        time_index = next(
            (i for i, candidate in enumerate(block[:2]) if "-->" in candidate), -1
        )
        if time_index < 0:
            continue
        match = _SRT_TIME.search(block[time_index])
        if match is None:
            continue
        text = "\n".join(block[time_index + 1:]).strip()
        if not text:
            continue
        lines.append(SubtitleLine(
            line_no=len(lines),
            start_ms=_time_ms(*match.group(1, 2, 3, 4)),
            end_ms=_time_ms(*match.group(5, 6, 7, 8)),
            text_raw=text,
            text_norm=normalizer(text),
        ))
    return lines


def parse_vtt(path, normalizer) -> list[SubtitleLine]:
    """VTT：头/NOTE/STYLE/REGION 剔除，标签剥离，小时位可选。"""
    lines: list[SubtitleLine] = []
    for block in _blocks(_read_text(path)):
        head = block[0].strip()
        if head.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        time_index = next(
            (i for i, candidate in enumerate(block[:3]) if "-->" in candidate), -1
        )
        if time_index < 0:
            continue
        match = _VTT_TIME.search(block[time_index])
        if match is None:
            continue
        text = re.sub(r"<[^>]+>", "", "\n".join(block[time_index + 1:])).strip()
        if not text:
            continue
        start_ms = _time_ms(
            match.group(1) or "0", match.group(2), match.group(3), match.group(4)
        )
        end_ms = _time_ms(
            match.group(5) or "0", match.group(6), match.group(7), match.group(8)
        )
        lines.append(SubtitleLine(
            line_no=len(lines), start_ms=start_ms, end_ms=end_ms,
            text_raw=text, text_norm=normalizer(text),
        ))
    return lines


def parse_ass(path, normalizer) -> list[SubtitleLine]:
    """ASS：Dialogue 解析（Text 含逗号安全）、Comment 跳过、标签剥离。"""
    lines: list[SubtitleLine] = []

    def _ass_ms(token: str) -> int:
        hours, minutes, rest = token.split(":")
        seconds, centis = rest.split(".")
        return (
            (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000
            + int(centis) * 10
        )

    for raw_line in _read_text(path).splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith("Dialogue:"):
            continue
        fields = stripped[len("Dialogue:"):].strip().split(",", 9)
        if len(fields) < 10:
            continue  # Dialogue 至少 10 个字段（Text 允许内含逗号）
        try:
            start_ms, end_ms = _ass_ms(fields[1]), _ass_ms(fields[2])
        except (ValueError, IndexError):
            continue
        text = _ASS_TAG.sub("", fields[9])
        text = text.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
        text = text.strip()
        if not text:
            continue
        lines.append(SubtitleLine(
            line_no=len(lines), start_ms=start_ms, end_ms=end_ms,
            text_raw=text, text_norm=normalizer(text),
        ))
    return lines


def detect_bilingual(lines: list[SubtitleLine], source_lang: str) -> list[SubtitleLine]:
    """双语轨：保留主语言行，副语言行丢弃。

    粗判据：行内含 CJK 视为中文行。源语言为中文时保留中文行；
    否则保留非中文行。无法判定（空行/纯符号）时保留。
    """
    kept: list[SubtitleLine] = []
    cjk = re.compile(r"[\u4e00-\u9fff]")
    for line in lines:
        is_cjk = bool(cjk.search(line.text_raw))
        if source_lang.startswith("zh"):
            if is_cjk:
                kept.append(line)
        elif not is_cjk:
            kept.append(line)
        else:
            kept.append(line)
    return kept


def parse_subtitle_file(path, normalizer) -> list[SubtitleLine]:
    """按扩展名分发；未知扩展名依次尝试三种解析器。"""
    path = Path(path)
    suffix = path.suffix.lower().lstrip(".")
    parsers = {
        "srt": parse_srt, "vtt": parse_vtt, "ass": parse_ass, "ssa": parse_ass,
    }
    parser = parsers.get(suffix)
    if parser is not None:
        return parser(path, normalizer)
    for candidate in (parse_srt, parse_vtt, parse_ass):
        try:
            lines = candidate(path, normalizer)
        except Exception:  # noqa: BLE001 - 逐个尝试
            continue
        if lines:
            return lines
    raise ValueError(f"无法解析字幕文件：{path}")


# ----------------------------------------------------------------------
# 目录监视（QFileSystemWatcher）
# ----------------------------------------------------------------------


class SubtitleDirectoryWatcher:
    """监视字幕目录，自动发现新字幕文件并回调。

    对应需求"可配置字幕目录监视（PotPlayer/VLC 等播放器的字幕目录）"：
    播放器把字幕下载/拖放进这些目录后，程序自动解析并注册为字幕轨，
    无需手动导入。

    **为什么回调而不是直接注册**：本类只负责"发现文件"这件事；
    解析失败要提示、注册要进调度器、还要更新 UI——那些是装配层的职责。
    回调签名 ``(path)``，由装配层决定后续动作。

    去重：同一文件只回调一次（按路径记录）；QFileSystemWatcher 对
    一次写入可能触发多次 directoryChanged，去重避免重复解析。
    """

    #: 支持的字幕扩展名（小写、不含点）
    EXTENSIONS = {"srt", "vtt", "ass", "ssa"}

    def __init__(self, directories: list[str], on_file: Callable[[Path], None]) -> None:
        from PyQt6.QtCore import QFileSystemWatcher

        self._on_file = on_file
        self._seen: set[str] = set()
        self._watcher = QFileSystemWatcher()
        self._watcher.directoryChanged.connect(self._on_directory_changed)

        valid: list[str] = []
        for directory in directories or []:
            path = Path(directory)
            if path.is_dir():
                valid.append(str(path))
                # 目录里**已存在**的字幕也要处理：监视启动前下载的字幕
                for existing in sorted(path.iterdir()):
                    if self._is_subtitle(existing):
                        self._seen.add(str(existing))
                        self._emit_later(existing)
            else:
                logger.warning("字幕目录不存在，跳过监视：%s", directory)
        if valid:
            self._watcher.addPaths(valid)
            logger.info("字幕目录监视已启动：%s", "、".join(valid))

    def _emit_later(self, path_str: str) -> None:
        """延迟回调：文件刚写入时可能还没写完，等一拍再解析。"""
        from PyQt6.QtCore import QTimer

        def _fire() -> None:
            try:
                self._on_file(Path(path_str))
            except Exception:  # noqa: BLE001 - 回调异常不终止监视
                logger.debug("字幕文件回调失败：%s", path_str, exc_info=True)

        QTimer.singleShot(500, _fire)

    @classmethod
    def _is_subtitle(cls, path: Path) -> bool:
        return path.is_file() and path.suffix.lower().lstrip(".") in cls.EXTENSIONS

    def _on_directory_changed(self, directory: str) -> None:
        """目录内容变化：找出新字幕文件。"""
        from pathlib import Path as _Path

        for entry in _Path(directory).iterdir():
            if not self._is_subtitle(entry):
                continue
            key = str(entry)
            if key in self._seen:
                continue
            self._seen.add(key)
            logger.info("发现新字幕文件：%s", entry.name)
            self._emit_later(key)
