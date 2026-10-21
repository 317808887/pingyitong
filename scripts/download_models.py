#!/usr/bin/env python
"""Spark-X2.5 模型权重下载脚本。

从 Hugging Face（或国内镜像）下载 Spark-X2.5 的 GGUF 量化权重到项目的
``models/`` 目录，供屏译通加载。

**只使用 Python 标准库**，因此无需先安装项目依赖即可执行 —— 这是刻意的：
用户克隆代码后第一件事就是下载模型，此时还没有 ``pip install``。

用法::

    # 下载两个模型（默认，约 3.45 GB）
    python scripts/download_models.py

    # 只下载极速模型（约 1.03 GB，适合 8GB 内存机器）
    python scripts/download_models.py --model 1.7b

    # 使用官方源（默认使用国内镜像 hf-mirror，因为 huggingface.co 在国内常不可达）
    python scripts/download_models.py --source hf

    # 校验已下载文件的完整性
    python scripts/download_models.py --verify-only

    # 查看下载目标位置
    python scripts/download_models.py --show-paths

下载支持**断点续传**：中途中断后重新执行会从已下载的字节继续，
不会重新开始。所有文件都会做 SHA256 校验，校验失败的会被删除并提示重试。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 分块下载的块大小（1 MiB）
CHUNK_SIZE = 1024 * 1024

#: 每个模型对应的仓库、文件名、大小与 SHA256。
#:
#: 大小与哈希取自 Hugging Face API（``/api/models/<repo>?blobs=true``），
#: 用于校验下载完整性；下载完成后会逐字节比对。
MODELS: dict[str, dict[str, object]] = {
    "1.7b": {
        "label": "Spark-X2.5-1.7B（极速模式）",
        "repo": "XHToken/Spark-X2.5-1.7B-GGUF",
        "filename": "Spark-X2.5-1.7B-Q4_K_M.gguf",
        "size": 1_107_457_856,
        "sha256": "902bde2522394954ac17821b3e5fd0df02defbc6944f122253f2580acf0503f4",
    },
    "4b": {
        "label": "Spark-X2.5-4B（高精度模式）",
        "repo": "XHToken/Spark-X2.5-4B-GGUF",
        "filename": "Spark-X2.5-4B-Q4_K_M.gguf",
        "size": 2_600_224_352,
        "sha256": "adfcfa19a4ed6a5985da8bf565fe15f8e1a7e131d79bae2d19d48d1c40109428",
    },
}

#: 下载源。默认使用 hf-mirror.com —— huggingface.co 在国内多数网络下不可达，
#: 而镜像是官方内容的完整同步，文件哈希一致。
SOURCES: dict[str, str] = {
    "hf-mirror": "https://hf-mirror.com",
    "hf": "https://huggingface.co",
}


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------


def project_models_dir() -> Path:
    """项目内的模型目录（随源码交付，优先使用）。"""
    return ROOT / "models"


def appdata_models_dir() -> Path:
    """应用数据目录中的模型目录（用户级安装时的位置）。"""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return Path(base) / "PingYiTong" / "models"
    return Path.home() / ".pingyitong" / "models"


def target_dir() -> Path:
    """返回默认下载目标目录。

    优先项目内的 ``models/``：这样源码目录自成一体，直接运行即可用，
    也方便打包分发。若项目目录不可写则退回应用数据目录。
    """
    candidate = project_models_dir()
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_test"
        probe.write_bytes(b"")
        probe.unlink()
        return candidate
    except OSError:
        fallback = appdata_models_dir()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def human_size(num_bytes: float) -> str:
    """把字节数格式化为易读形式。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """把秒数格式化为 ``1h23m`` 或 ``4m05s``。"""
    if seconds <= 0 or seconds != seconds or seconds == float("inf"):
        return "--"
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def check_url(url: str, timeout: float = 20.0) -> tuple[bool, str]:
    """探测 URL 是否可访问。返回 ``(是否可用, 说明)``。"""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return True, human_size(int(length)) if length else "未知大小"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - 网络异常种类繁多
        return False, type(exc).__name__


# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------


@dataclass
class DownloadResult:
    """一次下载的结果。"""

    key: str
    path: Path
    downloaded: int
    skipped: bool = False
    verified: bool = False
    error: str = ""


def download_file(
    url: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> DownloadResult:
    """带断点续传的下载，并在完成后校验完整性。

    :param url: 下载地址。
    :param destination: 目标文件路径。
    :param expected_size: 期望的字节数；``<= 0`` 表示大小未知，跳过大小校验。
    :param expected_sha256: 期望的 SHA256（小写十六进制）；
        空字符串表示跳过哈希校验。
    :param label: 用于进度显示的模型名称。
    """
    partial = destination.with_suffix(destination.suffix + ".part")
    check_size = expected_size > 0
    check_hash = bool(expected_sha256)

    # 已存在且完整：直接校验
    if destination.exists() and (not check_size or destination.stat().st_size == expected_size):
        print("  文件已存在且大小正确，校验哈希…" if check_hash else "  文件已存在，跳过下载")
        if not check_hash:
            return DownloadResult(label, destination, 0, skipped=True, verified=True)
        if verify_file(destination, expected_sha256):
            return DownloadResult(label, destination, 0, skipped=True, verified=True)
        print("  哈希不匹配，将重新下载")
        destination.unlink()

    # 断点续传：从 .part 文件继续
    resume_from = partial.stat().st_size if partial.exists() else 0
    # 只有已知总大小时才能判断 .part 是否已经完整
    if check_size and resume_from >= expected_size:
        resume_from = 0
        partial.unlink(missing_ok=True)

    headers = {"User-Agent": "PingYiTong-Downloader/1.0"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        print(f"  断点续传：从 {human_size(resume_from)} 继续")

    request = urllib.request.Request(url, headers=headers)
    downloaded = resume_from
    started = time.monotonic()

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            # 服务器不支持 Range 时会返回 200 而非 206，此时需从头开始
            if resume_from and response.status != 206:
                print("  服务器不支持断点续传，从头下载")
                resume_from = 0
                downloaded = 0

            # 大小未知时尽量从响应头拿到总长度，用于进度百分比
            reported = response.headers.get("Content-Length")
            total = expected_size
            if not check_size and reported:
                total = int(reported) + resume_from

            mode = "ab" if resume_from else "wb"
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    _print_progress(downloaded, total, started)
    except urllib.error.HTTPError as exc:
        _print_progress_end()
        return DownloadResult(label, destination, downloaded, error=f"HTTP {exc.code} {exc.reason}")
    except Exception as exc:  # noqa: BLE001 - 网络中断、磁盘错误等
        _print_progress_end()
        return DownloadResult(label, destination, downloaded, error=f"{type(exc).__name__}: {exc}")

    _print_progress_end()

    if check_size and downloaded != expected_size:
        return DownloadResult(
            label, destination, downloaded,
            error=f"大小不符：期望 {expected_size} 字节，实际 {downloaded} 字节",
        )

    if check_hash:
        # 校验通过才改名为正式文件，避免半成品被程序误加载
        print("  校验 SHA256…", end="", flush=True)
        if not verify_file(partial, expected_sha256):
            print(" 失败")
            return DownloadResult(
                label, destination, downloaded,
                error="SHA256 校验失败，文件可能在传输中损坏，请重试",
            )
        print(" 通过")

    partial.replace(destination)
    return DownloadResult(label, destination, downloaded, verified=True)


def _print_progress(done: int, total: int, started: float) -> None:
    """打印单行进度。"""
    ratio = done / total if total else 0.0
    elapsed = time.monotonic() - started
    speed = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / speed if speed > 0 else 0.0

    bar_width = 32
    filled = int(bar_width * ratio)
    bar = "█" * filled + "░" * (bar_width - filled)

    sys.stdout.write(
        f"\r  [{bar}] {ratio * 100:5.1f}%  "
        f"{human_size(done)} / {human_size(total)}  "
        f"{human_size(speed)}/s  剩余 {format_duration(remaining)}   "
    )
    sys.stdout.flush()


def _print_progress_end() -> None:
    """结束进度行。"""
    sys.stdout.write("\r" + " " * 110 + "\r")
    sys.stdout.flush()


def verify_file(path: Path, expected_sha256: str) -> bool:
    """计算文件 SHA256 并与期望值比对。

    分块读取，避免把数 GB 的文件一次性读入内存。
    """
    if not path.exists():
        return False

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return False

    return digest.hexdigest().lower() == expected_sha256.lower()


# ---------------------------------------------------------------------------
# GGUF 文件头检查
# ---------------------------------------------------------------------------

#: GGUF 文件魔数
GGUF_MAGIC = b"GGUF"

#: 值得展示的元数据键（GGUF 的建表键名）
_GGUF_INTERESTING_KEYS = (
    "general.architecture",
    "general.name",
    "general.quantization_version",
    "general.file_type",
)


def inspect_gguf(path: Path) -> dict[str, object]:
    """读取 GGUF 文件头与关键元数据。

    GGUF 的结构为：魔数 ``GGUF`` + 版本 + 张量数 + 元数据键值对数 + 元数据。
    这里只解析到元数据段的前若干项，足以确认"这是一个有效的 GGUF 模型文件"
    并读出架构信息，不需要完整解析数十万条张量描述。

    :return: ``{"valid": bool, "version": int, "tensors": int, "metadata": {...}}``
    """
    result: dict[str, object] = {"valid": False, "version": 0, "tensors": 0, "metadata": {}}

    try:
        with path.open("rb") as handle:
            if handle.read(4) != GGUF_MAGIC:
                return result

            version = int.from_bytes(handle.read(4), "little")
            tensor_count = int.from_bytes(handle.read(8), "little")
            kv_count = int.from_bytes(handle.read(8), "little")

            result["version"] = version
            result["tensors"] = tensor_count

            metadata: dict[str, object] = {}
            for _ in range(min(kv_count, 64)):
                key = _read_gguf_string(handle)
                if key is None:
                    break
                value = _read_gguf_value(handle)
                if value is _UNSUPPORTED:
                    break
                metadata[key] = value

            result["metadata"] = metadata
            result["valid"] = True
    except (OSError, ValueError):
        return result

    return result


_UNSUPPORTED = object()


def _read_gguf_string(handle) -> str | None:
    """读取 GGUF 字符串（长度为 uint64 前缀）。"""
    raw_length = handle.read(8)
    if len(raw_length) < 8:
        return None
    length = int.from_bytes(raw_length, "little")
    if length > 4096:  # 异常长度，说明文件已损坏
        return None
    if length == 0:
        return ""
    data = handle.read(length)
    if len(data) < length:
        return None
    return data.decode("utf-8", errors="replace")


def _read_gguf_value(handle):
    """读取 GGUF 元数据值（按类型标签分派）。"""
    tag_raw = handle.read(4)
    if len(tag_raw) < 4:
        return _UNSUPPORTED
    tag = int.from_bytes(tag_raw, "little")

    # 标量类型标签 → (字节数, 是否有符号)
    scalars = {
        0: (1, False), 1: (1, True),      # uint8 / int8
        2: (2, False), 3: (2, True),      # uint16 / int16
        4: (4, False), 5: (4, True),      # uint32 / int32
        6: (4, None),                     # float32
        7: (1, None),                     # bool
        10: (8, False), 11: (8, True),    # uint64 / int64
        12: (8, None),                    # float64
    }

    if tag == 8:  # string
        return _read_gguf_string(handle)

    if tag == 9:  # array：元素类型 + 元素个数 + 数据
        element_tag_raw = handle.read(4)
        count_raw = handle.read(8)
        if len(element_tag_raw) < 4 or len(count_raw) < 8:
            return _UNSUPPORTED
        count = int.from_bytes(count_raw, "little")
        if count > 100_000:  # 数组过大，跳过（例如分词表）
            return _UNSUPPORTED
        element_tag = int.from_bytes(element_tag_raw, "little")
        if element_tag == 8:
            return [_read_gguf_string(handle) for _ in range(min(count, 16))]
        size, _ = scalars.get(element_tag, (0, None))
        if size == 0:
            return _UNSUPPORTED
        handle.read(size * count)
        return f"<{count} 项数组>"

    size, signed = scalars.get(tag, (0, None))
    if size == 0:
        return _UNSUPPORTED

    raw = handle.read(size)
    if len(raw) < size:
        return _UNSUPPORTED

    if tag in (6, 12):  # float32 / float64
        import struct

        fmt = "<f" if tag == 6 else "<d"
        return struct.unpack(fmt, raw)[0]
    if tag == 7:  # bool
        return raw[0] != 0
    return int.from_bytes(raw, "little", signed=bool(signed))


def report_inspection(path: Path) -> None:
    """打印单个模型的 GGUF 头部检查结果。"""
    info = inspect_gguf(path)
    print(f"\n{path.name}")
    print(f"  大小：{human_size(path.stat().st_size)}")

    if not info["valid"]:
        print("  ✗ 不是有效的 GGUF 文件（魔数校验失败），文件可能已损坏")
        return

    print(f"  ✓ GGUF 有效（格式版本 {info['version']}，张量数 {info['tensors']:,}）")

    metadata = info["metadata"]
    assert isinstance(metadata, dict)
    for key in _GGUF_INTERESTING_KEYS:
        if key in metadata:
            print(f"  · {key} = {metadata[key]}")

    # 分词器规模可以侧面反映模型是否完整
    for key in ("tokenizer.ggml.tokens",):
        if key in metadata:
            print(f"  · {key} = {metadata[key]}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def ensure_source(source: str, repo: str, filename: str) -> tuple[str, str]:
    """确认下载源可用，返回 ``(基础地址, 说明)``。

    若首选源不可用，会按顺序尝试其余源，避免用户在网络受限时卡住。
    """
    order = [source] + [name for name in SOURCES if name != source]

    for name in order:
        base = SOURCES[name]
        url = f"{base}/{repo}/resolve/main/{filename}"
        ok, detail = check_url(url)
        if ok:
            if name != source:
                print(f"  首选源不可用，已切换到 {name}（{base}）")
            return base, name
        print(f"  源 {name} 不可用（{detail}），尝试下一个…")

    raise SystemExit(
        "错误：所有下载源均不可用。请检查网络连接，"
        "或手动从以下地址下载模型后放入 models/ 目录：\n"
        + "\n".join(f"  · https://{SOURCES[n]}/{repo}" for n in SOURCES)
    )


def download_model(key: str, source: str, target: Path, verify_only: bool = False) -> DownloadResult:
    """下载（或校验）单个模型。"""
    spec = MODELS[key]
    label = str(spec["label"])
    filename = str(spec["filename"])
    size = int(spec["size"])  # type: ignore[arg-type]
    sha256 = str(spec["sha256"])
    repo = str(spec["repo"])
    destination = target / filename

    print(f"\n{label}")
    print(f"  仓库：{repo}")
    print(f"  文件：{filename}  ({human_size(size)})")
    print(f"  目标：{destination}")

    if verify_only:
        if not destination.exists():
            return DownloadResult(label, destination, 0, error="文件不存在")
        print("  校验 SHA256…", end="", flush=True)
        if verify_file(destination, sha256):
            print(" 通过")
            return DownloadResult(label, destination, 0, skipped=True, verified=True)
        print(" 失败")
        return DownloadResult(label, destination, 0, error="SHA256 校验失败")

    base, _ = ensure_source(source, repo, filename)
    url = f"{base}/{repo}/resolve/main/{filename}"
    return download_file(url, destination, size, sha256, label)


def print_paths() -> None:
    """打印模型目录位置与查找规则。"""
    print("模型目录查找顺序（先找到的优先）：")
    print(f"  1. 项目内       {project_models_dir()}")
    print(f"  2. 应用数据目录 {appdata_models_dir()}")
    print()
    print(f"本次下载目标：{target_dir()}")
    print()
    print("程序按文件名识别模型，支持大小写不敏感匹配。推荐文件名：")
    for spec in MODELS.values():
        print(f"  · {spec['filename']}")


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="下载 Spark-X2.5 GGUF 模型权重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["1.7b", "4b", "all"], default="all",
        help="要下载的模型：1.7b（1.03GB）/ 4b（2.42GB）/ all（默认）",
    )
    parser.add_argument(
        "--source", choices=list(SOURCES), default="hf-mirror",
        help="下载源，默认 hf-mirror（国内镜像）",
    )
    parser.add_argument(
        "--target", type=Path, default=None,
        help="自定义下载目录（默认项目内 models/）",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="只校验已下载文件的完整性（SHA256），不下载",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="读取已下载文件的 GGUF 头部，确认是否为有效的模型文件",
    )
    parser.add_argument(
        "--show-paths", action="store_true",
        help="显示模型目录位置与文件名约定",
    )
    args = parser.parse_args(argv)

    if args.show_paths:
        print_paths()
        return 0

    target = args.target or target_dir()
    target.mkdir(parents=True, exist_ok=True)

    keys = list(MODELS) if args.model == "all" else [args.model]

    if args.inspect:
        # 只检查 GGUF 头部，不做哈希校验（速度更快，用于快速确认文件完整性）
        print("=" * 72)
        print("GGUF 文件头检查")
        print("=" * 72)
        print(f"检查目录：{target}")

        found_any = False
        for key in keys:
            filename = str(MODELS[key]["filename"])
            path = target / filename
            if not path.exists():
                # 也接受 .part 后缀的半成品，便于判断下载中断的位置
                partial = path.with_suffix(path.suffix + ".part")
                if partial.exists():
                    print(f"\n{filename}：尚未下载完成")
                    report_inspection(partial)
                    found_any = True
                else:
                    print(f"\n{filename}：文件不存在")
                continue
            report_inspection(path)
            found_any = True

        if not found_any:
            print("\n未找到任何模型文件。请先运行不带 --inspect 的命令进行下载。")
            return 1
        return 0

    print("=" * 72)
    print("屏译通 —— Spark-X2.5 模型权重下载")
    print("=" * 72)
    print(f"下载源：{SOURCES[args.source]}")
    print(f"目标目录：{target}")
    if not args.verify_only:
        total = sum(int(MODELS[k]["size"]) for k in keys)  # type: ignore[arg-type]
        print(f"总下载量：{human_size(total)}")

    results: list[DownloadResult] = []
    for key in keys:
        results.append(download_model(key, args.source, target, args.verify_only))

    # 汇总
    print()
    print("=" * 72)
    succeeded = [r for r in results if not r.error]
    failed = [r for r in results if r.error]

    for result in succeeded:
        state = "已存在并校验通过" if result.skipped else "下载并校验通过"
        print(f"  ✓ {result.path.name}（{state}）")
    for result in failed:
        print(f"  ✗ {result.path.name}：{result.error}")

    if failed:
        print()
        print("下载未全部完成。已下载的部分会保留，重新执行本脚本即可断点续传。")
        return 1

    print()
    print("全部完成。启动程序后会自动加载模型：")
    print("  python run.py")
    print()
    print("提示：程序启动时会自动加载模型（可在设置面板的「模型」分页关闭），")
    print("      首次加载 1.7B 约需数秒，4B 首次使用时会按需加载。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
