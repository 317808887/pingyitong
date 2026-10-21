"""日志系统。

提供按大小滚动的文件日志与控制台输出。

隐私约束：本应用为全离线工具，日志中不应出现屏幕识别文本或翻译结果。
调用方需自行遵守该约定（例如使用 :func:`describe_text` 记录文本长度而非内容）。
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from . import paths

_LOGGER_NAME = "pingyitong"
_configured = False

_LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_MAX_BYTES = 2 * 1024 * 1024
_BACKUP_COUNT = 3


def setup_logging(level: int = logging.INFO, console: bool = True) -> logging.Logger:
    """初始化根日志器。重复调用只生效一次。

    :param level: 文件日志级别。
    :param console: 是否同时输出到标准错误。
    """
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    try:
        log_file: Path = paths.log_dir() / "pingyitong.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # 数据目录不可写时降级为仅控制台输出，不阻断程序启动。
        pass

    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """获取子日志器。``name`` 通常传入 ``__name__``。"""
    if not _configured:
        setup_logging()
    if not name:
        return logging.getLogger(_LOGGER_NAME)
    short = name.split("pingyitong.")[-1]
    return logging.getLogger(f"{_LOGGER_NAME}.{short}")


def describe_text(text: str) -> str:
    """生成不含原文内容的文本描述，用于日志记录。

    仅返回字符数与首尾字符类型，避免屏幕内容写入日志文件。
    """
    if not text:
        return "empty"
    return f"{len(text)} chars"
