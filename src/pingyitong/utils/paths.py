"""文件系统路径解析。

统一管理应用数据目录、配置、日志、模型与资源文件的定位，
支持通过 ``PINGYITONG_HOME`` 环境变量或便携模式重定向数据目录。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..constants import APP_NAME_EN, ENV_HOME, ENV_MODELS_DIR

#: 便携模式标记：数据目录置于程序目录下的 ``data/`` 子目录。
_portable: bool = False


def set_portable(enabled: bool) -> None:
    """切换便携模式。应在其他路径函数首次调用前设置。"""
    global _portable
    _portable = bool(enabled)


def is_portable() -> bool:
    """返回当前是否处于便携模式。"""
    return _portable


def project_root() -> Path:
    """返回项目根目录。

    兼容两种运行形态：
    - 源码运行：``<root>/src/pingyitong/utils/paths.py`` → ``<root>``
    - 打包运行：可执行文件所在目录。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def bundled_dir() -> Path:
    """返回随程序分发的只读资源目录（``assets``）。

    打包后 PyInstaller 会将资源释放到 ``sys._MEIPASS``。
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "assets"
    return project_root() / "assets"


def app_data_dir() -> Path:
    """返回可写的应用数据目录，必要时自动创建。

    优先级：``PINGYITONG_HOME`` 环境变量 > 便携模式 > ``%LOCALAPPDATA%``。
    """
    override = os.environ.get(ENV_HOME)
    if override:
        path = Path(override).expanduser()
    elif _portable:
        path = project_root() / "data"
    else:
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        # 非 Windows 或环境变量缺失时的兜底
        path = Path(base) / APP_NAME_EN if base else Path.home() / f".{APP_NAME_EN.lower()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    """返回主配置文件路径。"""
    return app_data_dir() / "config.json"


def log_dir() -> Path:
    """返回日志目录，必要时自动创建。"""
    path = app_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """返回通用数据目录（历史数据库等），必要时自动创建。"""
    path = app_data_dir() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def history_db_path() -> Path:
    """返回翻译历史 SQLite 数据库路径。"""
    return data_dir() / "history.db"


def translation_cache_db_path() -> Path:
    """返回推理缓存持久化数据库路径。"""
    return data_dir() / "translation_cache.db"


def models_dir() -> Path:
    """返回默认的模型权重目录（应用数据目录下）。

    GGUF 模型体积较大，默认不随程序分发；但既可以从
    ``scripts/download_models.py`` 下载到此目录，也可以放在**项目内的
    ``models/``** 中（源码运行时更方便）。查找顺序见
    :func:`models_search_dirs`。
    """
    path = app_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_models_dir() -> Path:
    """返回项目内的模型目录（``<项目根>/models``）。

    源码方式运行时优先从这里加载，这样仓库自成一体、直接可用；
    打包后该目录通常不存在，会自动回退到应用数据目录。
    """
    return project_root() / "models"


def models_search_dirs() -> list[Path]:
    """返回模型文件的查找目录，按优先级从高到低排列。

    顺序为：环境变量覆盖目录 → 项目内 → 应用数据目录。

    之所以要支持多个位置：用户可能把权重放在仓库里（开发/评审场景），
    也可能放在应用数据目录（正式安装场景），程序应两者都能找到，
    而不是要求用户把文件搬到特定位置。
    """
    directories: list[Path] = []

    override = os.environ.get(ENV_MODELS_DIR)
    if override:
        directories.append(Path(override).expanduser())

    directories.append(project_models_dir())
    # app_data_dir() 已经处理了 PINGYITONG_HOME 与便携模式
    directories.append(app_data_dir() / "models")

    # 去重但保持顺序
    unique: list[Path] = []
    for directory in directories:
        if directory not in unique:
            unique.append(directory)
    return unique


def resolve_model_path(file_name: str) -> Path:
    """在全部候选目录中解析模型文件的实际路径。

    匹配策略由严到宽，逐级放宽：

    1. **精确文件名匹配**——配置里的名字与实际文件一致；
    2. **大小写不敏感匹配**——官方发布的文件名大小写与配置不同
       （如 ``Spark-X2.5-1.7B-Q4_K_M.gguf`` vs 小写形式）；
    3. **特征匹配**——目录里只放了一个同参数量的 GGUF，就采用它。
       用户从 Ollama / LM Studio 下载的文件名往往带有额外后缀
       （如 ``spark-x2.5-1.7b-q4_k_m-00001-of-00001.gguf``），
       不应因此加载失败。

    :param file_name: 配置中声明的文件名。
    :return: 解析到的路径。全部候选目录都找不到时，返回"应用数据目录 /
        该文件名"作为**期望路径**——调用方据此在错误提示中告诉用户
        应该把文件放到哪里，而不是给出一个莫须有的路径。
    """
    directories = models_search_dirs()
    stem = Path(file_name).stem.lower()

    # 1) 精确匹配
    for directory in directories:
        candidate = directory / file_name
        if candidate.is_file():
            return candidate

    # 2) 大小写不敏感匹配
    for directory in directories:
        if not directory.is_dir():
            continue
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_file() and entry.name.lower() == file_name.lower():
                return entry

    # 3) 特征匹配：从配置文件名中提取参数量标识（1.7b / 4b）后模糊匹配
    keyword = _model_keyword(stem)
    if keyword:
        for directory in directories:
            if not directory.is_dir():
                continue
            try:
                candidates = sorted(
                    entry
                    for entry in directory.iterdir()
                    if entry.is_file()
                    and entry.suffix.lower() == ".gguf"
                    and keyword in entry.name.lower()
                )
            except OSError:
                continue
            if candidates:
                return candidates[0]

    # 都不存在：返回期望路径，供错误提示使用
    return models_dir() / file_name


def _model_keyword(stem: str) -> str:
    """从模型文件名中提取参数量标识，用于模糊匹配。

    ``spark-x2.5-1.7b-q4_k_m`` → ``1.7b``；``spark-x2.5-4b-q4_k_m`` → ``4b``。
    """
    import re

    match = re.search(r"(\d+(?:\.\d+)?)b", stem)
    return f"{match.group(1)}b".lower() if match else ""


def model_path(file_name: str) -> Path:
    """返回指定模型文件的完整路径（已解析实际位置）。

    .. deprecated:: 0.5
        直接使用 :func:`resolve_model_path`。保留此函数是为了兼容
        早期调用点，二者行为一致。
    """
    return resolve_model_path(file_name)


def find_model_file(file_name: str) -> Path | None:
    """查找模型文件，找不到时返回 ``None``（不猜测路径）。

    适用于只想判断"是否存在"、不希望拿到猜测路径的场景。
    """
    resolved = resolve_model_path(file_name)
    return resolved if resolved.is_file() else None


def llama_runtime_candidates() -> list[Path]:
    """返回 ``llama-server`` 可执行文件的候选路径，按优先级排列。

    ``scripts/download_runtime.py`` 会把预编译的推理运行时解压到
    ``runtime/llama/``。
    """
    return [
        project_root() / "runtime" / "llama" / "llama-server.exe",
        project_root() / "llama-server.exe",
        app_data_dir() / "runtime" / "llama" / "llama-server.exe",
    ]


def llama_runtime_dir() -> Path:
    """返回推理运行时的安装目录（必要时创建）。"""
    path = project_root() / "runtime" / "llama"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ocr_models_dir() -> Path:
    """返回 OCR 模型缓存目录（RapidOCR 首次运行会自动下载）。"""
    path = app_data_dir() / "ocr_models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_glossary_path() -> Path:
    """返回用户自定义术语表默认路径。"""
    return data_dir() / "glossary.csv"


def user_style_path() -> Path:
    """返回用户翻译样式覆盖文件路径。"""
    return data_dir() / "style.json"


def asset_path(*parts: str) -> Path:
    """拼接并返回 ``assets`` 目录下的资源路径。"""
    return bundled_dir().joinpath(*parts)


def icon_path(file_name: str) -> Path:
    """返回图标文件路径。"""
    return asset_path("icons", file_name)


def default_config_path() -> Path:
    """返回随程序分发的默认配置文件路径。"""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "config" / "default_config.json"
    return project_root() / "config" / "default_config.json"


def sample_glossary_path() -> Path:
    """返回示例术语表路径。"""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "config" / "glossary_sample.csv"
    return project_root() / "config" / "glossary_sample.csv"


def builtin_glossary_path(domain: str = "software_ai") -> Path:
    """Return a shipped terminology snapshot; reject arbitrary path fragments."""
    from ..glossary_catalog import BUILTIN_GLOSSARY_DOMAINS

    if domain not in BUILTIN_GLOSSARY_DOMAINS:
        raise ValueError(f"未知预装术语领域：{domain}")
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "config" / "glossaries" / f"{domain}.json"
    return project_root() / "config" / "glossaries" / f"{domain}.json"
