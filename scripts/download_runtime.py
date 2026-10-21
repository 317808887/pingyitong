#!/usr/bin/env python
"""推理运行时下载脚本。

下载 **llama-server** 可执行文件到 ``runtime/llama/``，供屏译通调用。

为什么需要它
    Spark-X2.5 使用的 ``spark2_5`` 架构**只有 XHToken 维护的 llama.cpp 分支
    支持**；上游 ``ggml-org/llama.cpp``（也就是 ``llama-cpp-python`` 捆绑的
    那个）不认识该架构，加载时直接报::

        unknown model architecture: 'spark2_5'

    这不是版本问题，换任何版本的 ``llama-cpp-python`` 都无效。因此
    Spark-X2.5 必须走 ``llama-server`` 外部进程后端。

下载内容
    - ``llama-spark-x2.5-win-x64-cuda12.4-vulkan.zip``
      llama-server / llama-cli 等可执行文件，含 CPU、CUDA 12.4、Vulkan 后端
    - ``cuda-runtime-win-x64-12.4.zip``（可选，约 372 MB）
      CUDA 运行时 DLL，仅在使用 CUDA 后端时需要。
      **新显卡（如 RTX 50 系列的 Blackwell 架构）请改用 Vulkan 后端**，
      因为 CUDA 12.4 不支持 sm_120 计算能力；Vulkan 后端无需额外文件。

用法::

    python scripts/download_runtime.py                 # 下载主程序（约 245 MB）
    python scripts/download_runtime.py --with-cuda     # 同时下载 CUDA 运行时
    python scripts/download_runtime.py --verify-only   # 只检查已下载内容
    python scripts/download_runtime.py --show-paths    # 查看安装位置
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# 复用模型下载脚本里的通用工具，避免重复实现
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_models import (  # noqa: E402  # noqa: E402
    check_url,
    download_file,
    human_size,
)

ROOT = Path(__file__).resolve().parents[1]

#: 发布来源：由社区从 XHToken/llama.cpp 分支构建的 Windows x64 版本。
#: 构建脚本公开在仓库的 GitHub Actions 中，可自行复现。
RELEASE_REPO = "dmitrysdm/llama-Spark-x2.5-win-x64"

#: 直连地址与镜像前缀。GitHub 的 Release 资源在国内常被重置连接，
#: 因此默认走镜像；镜像不可用时回退直连。
MIRROR_PREFIXES: tuple[str, ...] = (
    "https://gh-proxy.com/",
    "https://ghfast.top/",
    "",  # 直连
)

ASSETS: dict[str, dict[str, object]] = {
    "runtime": {
        "label": "推理运行时（llama-server，含 CPU/CUDA/Vulkan 后端）",
        "filename": "llama-spark-x2.5-win-x64-cuda12.4-vulkan.zip",
        "required": True,
        # 字节数取自 GitHub Release API，用于校验下载完整性。
        # 发布方未提供 SHA256，因此完整性依赖 HTTPS + 大小校验 + zip 结构校验。
        "size": 256_981_915,
    },
    "cuda": {
        "label": "CUDA 运行时 DLL（仅 CUDA 后端需要；新显卡请用 Vulkan）",
        "filename": "cuda-runtime-win-x64-12.4.zip",
        "required": False,
        "size": 389_641_385,
    },
}


def runtime_dir() -> Path:
    """返回推理运行时的安装目录。"""
    return ROOT / "runtime" / "llama"


def asset_url(filename: str) -> str:
    """返回资源的 GitHub Release 直连地址。"""
    return (
        f"https://github.com/{RELEASE_REPO}/releases/download/latest/{filename}"
    )


def pick_mirror(url: str) -> str:
    """选择一个可用且速度可接受的镜像前缀。

    通过实测下载 2MB 数据来判断可达性与速度——只看 HTTP 状态码不够，
    部分镜像会返回 200 但速度极慢。
    """
    best_prefix = ""
    best_speed = 0.0

    for prefix in MIRROR_PREFIXES:
        candidate = f"{prefix}{url}"
        label = prefix or "直连"
        ok, detail = check_url(candidate, timeout=20.0)
        if not ok:
            print(f"  {label}：不可用（{detail}）")
            continue

        speed = _measure_speed(candidate)
        print(f"  {label}：可用，实测 {human_size(speed)}/s")
        if speed > best_speed:
            best_speed = speed
            best_prefix = prefix

    if best_speed <= 0:
        raise SystemExit(
            "错误：所有下载源均不可用。请检查网络，或手动从以下地址下载\n"
            f"  {asset_url(ASSETS['runtime']['filename'])}\n"
            f"并解压到 {runtime_dir()}"
        )

    return best_prefix


def _measure_speed(url: str, sample_bytes: int = 2 * 1024 * 1024) -> float:
    """实测下载速度（字节/秒）。失败返回 0。"""
    import time
    import urllib.request

    request = urllib.request.Request(
        url, headers={"Range": f"bytes=0-{sample_bytes - 1}", "User-Agent": "PingYiTong/1.0"}
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            received = 0
            while received < sample_bytes:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                received += len(chunk)
    except Exception:  # noqa: BLE001 - 测速失败按不可用处理
        return 0.0

    elapsed = time.monotonic() - started
    return received / elapsed if elapsed > 0 else 0.0


def extract(archive: Path, destination: Path) -> int:
    """解压归档到目标目录，返回解压出的文件数。"""
    destination.mkdir(parents=True, exist_ok=True)
    count = 0

    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            # 防御性检查：拒绝绝对路径与上级目录跳转
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                print(f"  跳过可疑条目：{info.filename}")
                continue
            bundle.extract(info, destination)
            count += 1

    return count


def verify_installation() -> bool:
    """检查运行时是否已正确安装。"""
    directory = runtime_dir()
    server = directory / "llama-server.exe"

    print(f"安装目录：{directory}")
    if not directory.exists():
        print("  ✗ 目录不存在")
        return False

    if not server.is_file():
        print("  ✗ 未找到 llama-server.exe")
        return False

    size = server.stat().st_size
    print(f"  ✓ llama-server.exe（{human_size(size)}）")

    # 列出关键后端 DLL，让用户知道有哪些加速方式可用
    for pattern, label in (
        ("ggml-cuda.dll", "CUDA 后端"),
        ("ggml-vulkan.dll", "Vulkan 后端"),
        ("ggml-cpu*.dll", "CPU 后端"),
    ):
        matches = sorted(directory.glob(pattern))
        if matches:
            print(f"  ✓ {label}：{', '.join(p.name for p in matches[:3])}")
        else:
            print(f"  · {label}：未找到")

    cudart = sorted(directory.glob("cudart64_*.dll")) + sorted(
        directory.glob("cublas64_*.dll")
    )
    if cudart:
        print(f"  ✓ CUDA 运行时：{', '.join(p.name for p in cudart[:3])}")
    else:
        print("  · CUDA 运行时：未安装（CUDA 后端不可用；Vulkan 与 CPU 不受影响）")

    return True


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="下载 llama-server 推理运行时（Spark-X2.5 必需）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "说明：Spark-X2.5 的模型架构只有 XHToken 维护的 llama.cpp 分支支持，\n"
            "      因此本运行时是让它真正跑起来的必要组件。\n"
        ),
    )
    parser.add_argument(
        "--with-cuda", action="store_true",
        help="同时下载 CUDA 运行时 DLL（约 372 MB，仅 CUDA 后端需要）",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="只检查已安装的运行时，不下载",
    )
    parser.add_argument(
        "--show-paths", action="store_true",
        help="显示安装位置",
    )
    parser.add_argument(
        "--keep-archives", action="store_true",
        help="保留下载的压缩包（默认解压后删除以节省空间）",
    )
    args = parser.parse_args(argv)

    if args.show_paths:
        print("推理运行时安装位置：")
        print(f"  {runtime_dir()}")
        print()
        print("程序查找 llama-server 的顺序：")
        print("  1. 环境变量 PINGYITONG_LLAMA_SERVER")
        print(f"  2. {runtime_dir()}\\llama-server.exe")
        print(f"  3. {ROOT}\\llama-server.exe")
        print("  4. 系统 PATH")
        return 0

    if args.verify_only:
        print("=" * 72)
        print("推理运行时检查")
        print("=" * 72)
        return 0 if verify_installation() else 1

    print("=" * 72)
    print("屏译通 —— 推理运行时下载（llama-server）")
    print("=" * 72)
    print("来源：社区构建的 XHToken/llama.cpp 分支 Windows 版本")
    print(f"仓库：https://github.com/{RELEASE_REPO}")
    print()

    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)

    targets = ["runtime"] + (["cuda"] if args.with_cuda else [])

    # 先确定镜像（只需测一次）
    print("选择下载源…")
    prefix = pick_mirror(asset_url(str(ASSETS["runtime"]["filename"])))
    print(f"使用：{prefix or '直连'}")
    print()

    failures: list[str] = []

    for key in targets:
        spec = ASSETS[key]
        filename = str(spec["filename"])
        label = str(spec["label"])

        print(label)
        print(f"  文件：{filename}")

        archive = directory / filename
        url = f"{prefix}{asset_url(filename)}"

        # 发布方未提供 SHA256，因此传空哈希跳过校验；
        # 完整性依赖 HTTPS + 大小校验 + 解压时的 zip 结构校验三重保障。
        result = download_file(
            url, archive, int(spec["size"]), "", label  # type: ignore[arg-type]
        )
        if result.error:
            print(f"  ✗ 下载失败：{result.error}")
            failures.append(filename)
            continue

        print(f"  解压到 {directory}")
        try:
            count = extract(archive, directory)
        except zipfile.BadZipFile:
            print("  ✗ 压缩包损坏，请重新运行本脚本")
            archive.unlink(missing_ok=True)
            failures.append(filename)
            continue

        print(f"  已解压 {count} 个文件")

        if not args.keep_archives:
            archive.unlink(missing_ok=True)
            print("  已删除压缩包（如需保留请加 --keep-archives）")
        print()

    print("=" * 72)
    if verify_installation():
        print()
        print("运行时安装完成。现在可以启动程序：")
        print("  python run.py")
        print()
        print("提示：程序启动时会自动拉起 llama-server 并加载模型；")
        print("      首次加载 1.7B 约需数秒。")
        return 0

    if failures:
        print(f"\n以下组件下载失败：{', '.join(failures)}")
    print("运行时未正确安装。可手动下载后解压到上面显示的目录：")
    print(f"  {asset_url(str(ASSETS['runtime']['filename']))}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
