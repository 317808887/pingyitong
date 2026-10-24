#!/usr/bin/env python
"""打包脚本。

把项目打包为独立可执行文件，并可选地生成 Windows 安装包。

用法::

    # 生成可执行文件到 build/dist/屏译通/
    python build/build_exe.py

    # 同时生成单文件版本（启动较慢，但便于分发）
    python build/build_exe.py --onefile

    # 生成安装包（需要安装 Inno Setup 6）
    python build/build_exe.py --installer

    # 只生成安装包（跳过已完成的 EXE 构建）
    python build/build_exe.py --installer --skip-build

依赖::

    pip install -r requirements-dev.txt

打包产物::

    build/dist/屏译通/屏译通.exe          可执行文件
    build/Output/屏译通-Setup-<版本>.exe   安装包
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

#: 项目根目录
ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
DIST_DIR = BUILD_DIR / "dist"
WORK_DIR = BUILD_DIR / "work"
OUTPUT_DIR = BUILD_DIR / "Output"

#: 应用显示名（与 constants.APP_NAME 保持一致）
APP_NAME = "屏译通"
APP_NAME_EN = "PingYiTong"


def validate_onedir_bundle(dist_app: Path) -> None:
    """Fail closed when the onedir output is incomplete or unusable.

    PyInstaller's ``COLLECT`` step can fail while removing a previous dist
    tree (for example, when a running application has a model file open).
    The old tree may still contain an EXE, so checking only for that EXE can
    otherwise let ``--skip-build`` publish a broken installer.
    """
    executable = dist_app / f"{APP_NAME_EN}.exe"
    internal_dir = dist_app / "_internal"
    runtime_archive = internal_dir / "base_library.zip"
    python_dlls = tuple(internal_dir.glob("python*.dll")) if internal_dir.is_dir() else ()

    missing = [
        str(path.relative_to(dist_app))
        for path in (executable, runtime_archive)
        if not path.is_file()
    ]
    if not python_dlls:
        missing.append("_internal/python*.dll")
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"目录版打包产物不完整，缺少：{missing_list}。"
            "请先关闭正在运行的屏译通，再重新执行完整构建；"
            "不要使用 --skip-build 生成安装包。"
        )

    try:
        with zipfile.ZipFile(runtime_archive) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"损坏条目：{bad_member}")
            members = set(archive.namelist())
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise RuntimeError(
            f"目录版打包产物中的 {runtime_archive.name} 无法通过 ZIP 完整性检查：{exc}。"
            "请重新执行完整构建。"
        ) from exc

    required_members = {"codecs.pyc", "encodings/__init__.pyc", "io.pyc"}
    missing_members = sorted(required_members - members)
    if missing_members:
        raise RuntimeError(
            f"{runtime_archive.name} 缺少 Python 标准库条目："
            f"{', '.join(missing_members)}。请重新执行完整构建。"
        )


def read_version() -> str:
    """从 version.py 读取版本号，避免在两处维护。

    直接解析源码而不导入模块，是为了让打包脚本不依赖项目的运行时依赖
    （PyQt6 等），这样在纯净环境中也能执行。
    """
    version_file = ROOT / "src" / "pingyitong" / "version.py"
    for line in version_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("__version__"):
            # 形如 __version__: Final[str] = "0.5.0"
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"无法从 {version_file} 解析版本号")


def check_dependencies(require_pyinstaller: bool = True) -> None:
    """检查打包所需依赖是否就绪。"""
    missing: list[str] = []

    if require_pyinstaller:
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            missing.append("pyinstaller")

    if missing:
        print("缺少打包依赖：", file=sys.stderr)
        for name in missing:
            print(f"  · {name}", file=sys.stderr)
        print(
            "\n请执行：python -m pip install -r requirements-dev.txt",
            file=sys.stderr,
        )
        raise SystemExit(2)


def find_inno_setup() -> Path | None:
    """定位 Inno Setup 的编译器 ``ISCC.exe``。

    按常见安装位置查找，因为 Inno Setup 默认不加入 PATH。
    """
    candidates = [
        # Windows 注册的环境变量名就是混大小写的，保持原样
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))  # noqa: SIM112
        / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))  # noqa: SIM112
        / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs" / "Inno Setup 6" / "ISCC.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def clean() -> None:
    """清理上次构建的产物。

    保留 ``dist/`` 之外的目录内容，避免误删用户放置的模型文件。
    """
    for directory in (DIST_DIR, WORK_DIR):
        if directory.exists():
            print(f"清理 {directory.relative_to(ROOT)}")
            try:
                shutil.rmtree(directory)
            except OSError as exc:
                raise RuntimeError(
                    f"无法清理旧构建目录 {directory}：{exc}。"
                    "请关闭正在运行的屏译通后重试，避免继续使用残缺的旧产物。"
                ) from exc


def build_executable(
    onefile: bool = False,
    console: bool = False,
    exclude_cuda: bool = True,
) -> Path:
    """调用 PyInstaller 构建可执行文件。

    :param onefile: 是否生成单文件。
    :param console: 是否保留控制台窗口（调试时有用）。
    :param exclude_cuda: 是否排除 CUDA 后端（``ggml-cuda.dll``，约 513MB）。
        默认排除：本项目随附的 CUDA 12.4 构建**不含 RTX 50 系（Blackwell
        sm_120）内核**，在这类显卡上加载即失败；Vulkan 后端在所有
        NVIDIA / AMD / Intel 显卡上都可用，因此排除 CUDA 既省体积
        又不损失可用性。需要为旧卡启用 CUDA 时传 ``--with-cuda``。
    :return: 生成的可执行文件路径。
    """
    import PyInstaller.__main__

    #: 需要随程序分发的数据文件：(源路径, 目标目录)
    #:
    #: 模型权重不在此列：体积达数 GB，且许可上要求用户自行获取，
    #: 因此由用户在应用数据目录中放置。
    datas = [
        (str(ROOT / "config"), "config"),
        (str(ROOT / "assets"), "assets"),
    ]

    #: 显式排除用不到的重型依赖，显著减小打包体积。
    #: 这些包可能是某些间接依赖的推荐项，但本项目并不使用。
    # 注意：**不要排除 PIL / pyclipper / shapely / yaml**——它们是
    # RapidOCR 的硬依赖（PIL 用于图像读写，pyclipper 用于文本框裁剪，
    # shapely 用于多边形运算）。曾因排除 PIL 导致打包版 OCR 整体不可用，
    # 表现为启动后提示"OCR 组件不可用：No module named ..."。
    excludes = [
        "tkinter", "matplotlib", "scipy", "pandas",
        "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngineWidgets", "PyQt6.QtQml",
        "PyQt6.QtQuick", "PyQt6.Qt3DCore", "PyQt6.QtMultimedia",
        "PyQt6.QtBluetooth", "PyQt6.QtDesigner", "PyQt6.QtHelp",
        "PyQt6.QtNetwork", "PyQt6.QtSql", "PyQt6.QtTest",
        "PyQt6.QtSensors", "PyQt6.QtSerialPort", "PyQt6.QtPositioning",
        "PyQt6.QtRemoteObjects", "PyQt6.QtTextToSpeech", "PyQt6.QtWebChannel",
        "PyQt6.QtWebSockets", "PyQt6.QtNfc", "PyQt6.QtSvg",
        # 测试与开发工具（setuptools/pkg_resources 不排除：
        # 部分科学计算库在导入期会引用它们，排除后表现为随机 ImportError）
        "pytest", "unittest", "pydoc", "doctest",
    ]

    arguments = [
        str(ROOT / "run.py"),
        # 可执行文件名使用英文：非 ASCII 文件名在部分构建环境与杀毒软件
        # 白名单处理中容易出问题。界面显示名仍为中文（见 constants.APP_NAME）。
        "--name", APP_NAME_EN,
        "--distpath", str(DIST_DIR),
        "--workpath", str(WORK_DIR),
        "--specpath", str(WORK_DIR),
        "--noconfirm",
        "--clean",
        # 图标：使用 PPT 封面展示的软件图标
        "--icon", str(ROOT / "assets" / "icons" / "app_icon.ico"),
        # 源码目录需加入搜索路径，因为 run.py 依赖 src 布局
        "--paths", str(ROOT / "src"),
    ]

    if onefile:
        arguments.append("--onefile")
    else:
        arguments.append("--onedir")

    arguments.append("--console" if console else "--windowed")

    for source, target in datas:
        arguments.extend(["--add-data", f"{source}{os.pathsep}{target}"])

    for module in excludes:
        arguments.extend(["--exclude-module", module])

    # 隐藏导入：这些模块通过字符串或协议动态引用，PyInstaller 的静态分析
    # 无法发现，必须显式声明，否则打包后运行时报 ImportError。
    hidden_imports = [
        "mss",
        "cv2",
        "psutil",
        "winrt.windows.networking.connectivity",
        "winrt.windows.foundation",
        "winrt.windows.foundation.collections",
    ]
    for module in hidden_imports:
        arguments.extend(["--hidden-import", module])

    # 这些包必须用 --collect-all 而不是 --hidden-import：
    # 它们既有**动态导入的子模块**（PyInstaller 静态分析看不到），
    # 又有**运行时按路径加载的数据文件**，缺任一部分都会在运行时报
    # ImportError 或文件找不到。
    #
    # 实测教训：只写 --hidden-import rapidocr_onnxruntime 时，包被
    # 漏收，程序启动后报 "OCR 组件不可用：No module named 'rapidocr'"
    # ——那条消息其实来自代码里的降级分支（`from rapidocr import ...`），
    # 真正的原因是 rapidocr_onnxruntime 没能导入。
    collect_all = [
        # RapidOCR：子模块 ch_ppocr_det/rec/cls + models/*.onnx（15.6MB）
        "rapidocr_onnxruntime",
        # trafilatura：正文抽取规则与配置文件
        "trafilatura",
        # charset-normalizer：编码探测所需的数据表
        "charset_normalizer",
    ]
    for package in collect_all:
        arguments.extend(["--collect-all", package])

    print("正在打包可执行文件…")
    print("  入口：run.py")
    print(f"  版本：{read_version()}")
    print(f"  模式：{'单文件' if onefile else '目录'}")
    print()

    PyInstaller.__main__.run(arguments)

    dist_app = DIST_DIR if onefile else DIST_DIR / APP_NAME_EN
    if not onefile:
        # Verify the runtime archive at the first boundary after PyInstaller
        # has collected files, before copying assets or producing an installer.
        validate_onedir_bundle(dist_app)

    # PyInstaller 会扫描构建进程的 PATH 来解析 DLL。Codex 工作区的
    # PATH 中带有 Poppler，其 ICU DLL 会被误收到 ``_internal`` 根目录。
    # Windows 加载器随后会让这两个文件遮蔽 System32 中与 Qt 6.11
    # ABI 匹配的系统 ICU，最终导致 ``PyQt6.QtCore`` 报 DLL load failed。
    # 它们不是本项目依赖，且 Qt 在源码环境本来就使用系统 ICU，
    # 因此只清理产物根目录下这两个已知的环境污染文件。
    internal_dir = dist_app / "_internal"
    for conflict_name in ("icuuc.dll", "icudt78.dll"):
        conflict = internal_dir / conflict_name
        if conflict.is_file():
            conflict.unlink()
            print(f"  已移除构建环境污染 DLL：{conflict_name}")

    # The installer creates a user-manual shortcut. Ship its actual target
    # beside the EXE so that the installed shortcut can open the manual.
    if not onefile:
        for relative in ("README.md", "docs/用户手册.md", "docs/架构说明.md"):
            document = ROOT / relative
            target_document = dist_app / relative
            target_document.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(document, target_document)

    # 复制推理运行时和模型权重到 dist 目录（安装包一并打入）
    cuda_patterns = ["ggml-cuda.dll", "ggml-cuda*.dll", "cudart*", "cublas*"] if exclude_cuda else []
    for label, src, dst_name, extra_ignore in (
        ("推理运行时", ROOT / "runtime" / "llama", "runtime/llama", cuda_patterns),
        ("模型权重", ROOT / "models", "models", []),
    ):
        dst = dist_app / dst_name
        if not src.exists():
            print(f"  跳过 {label}（不存在：{src}）")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        print(f"  复制 {label} …" + ("（已排除 CUDA 后端）" if extra_ignore else ""))
        shutil.copytree(
            str(src), str(dst),
            ignore=shutil.ignore_patterns("*.log", *extra_ignore),
        )

    if onefile:
        return dist_app / f"{APP_NAME_EN}.exe"

    return dist_app / f"{APP_NAME_EN}.exe"


def build_installer(skip_build: bool = False) -> Path | None:
    """调用 Inno Setup 生成安装包。

    :param skip_build: ``True`` 时跳过 EXE 构建，直接打包已有产物。
    :return: 生成的安装包路径；未找到 Inno Setup 时返回 ``None``。
    """
    dist_app = DIST_DIR / APP_NAME_EN
    try:
        # This check is required even with --skip-build: a stale EXE can remain
        # after COLLECT fails, while base_library.zip is absent from dist.
        validate_onedir_bundle(dist_app)
    except RuntimeError as exc:
        print(f"\n无法生成安装包：{exc}", file=sys.stderr)
        return None

    iscc = find_inno_setup()
    if iscc is None:
        print(
            "\n未找到 Inno Setup 6（ISCC.exe）。\n"
            "请从 https://jrsoftware.org/isdl.php 下载安装，"
            "或使用 --skip-build 之外的参数仅生成可执行文件。",
            file=sys.stderr,
        )
        return None

    if not skip_build:
        executable = dist_app / f"{APP_NAME_EN}.exe"
        if not executable.exists():
            print(
                f"未找到已构建的可执行文件：{executable}\n"
                "请先运行不带 --skip-build 的命令。",
                file=sys.stderr,
            )
            return None

    script = BUILD_DIR / "installer.iss"
    if not script.exists():
        print(f"未找到 Inno Setup 脚本：{script}", file=sys.stderr)
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    version = read_version()
    print("\n正在生成安装包…")
    result = subprocess.run(
        [
            str(iscc),
            f"/DMyAppVersion={version}",
            f"/DMySourceDir={DIST_DIR / APP_NAME_EN}",
            f"/DMyOutputDir={OUTPUT_DIR}",
            str(script),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        print("Inno Setup 编译失败：", file=sys.stderr)
        print(result.stdout[-4000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        return None

    installer = OUTPUT_DIR / f"{APP_NAME}-Setup-{version}.exe"
    return installer if installer.exists() else None


def report(path: Path | None, label: str) -> None:
    """输出产物信息。"""
    if path is None or not path.exists():
        return
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  {label}：{path.relative_to(ROOT)}  （{size_mb:.1f} MB）")


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="打包屏译通为可执行文件与安装包",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--onefile", action="store_true",
        help="生成单文件可执行程序（启动较慢，便于分发）",
    )
    parser.add_argument(
        "--console", action="store_true",
        help="保留控制台窗口（调试打包问题时使用）",
    )
    parser.add_argument(
        "--installer", action="store_true",
        help="生成 Inno Setup 安装包",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="跳过可执行文件构建，直接用已有产物生成安装包",
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="不清理上次构建的中间产物",
    )
    parser.add_argument(
        "--with-cuda", action="store_true",
        help="包含 CUDA 后端（+513MB；RTX 50 系不可用，仅旧卡需要）",
    )
    args = parser.parse_args(argv)

    print("屏译通 打包工具")
    print(f"项目目录：{ROOT}")
    print()

    if not (ROOT / "src" / "pingyitong").exists():
        print(f"错误：未找到源码目录 {ROOT / 'src' / 'pingyitong'}", file=sys.stderr)
        return 2

    if sys.platform != "win32":
        print(
            "警告：当前不是 Windows 平台。可执行文件需在目标平台上构建，"
            "跨平台打包不会得到可用的 Windows 程序。",
            file=sys.stderr,
        )

    check_dependencies(require_pyinstaller=not args.skip_build)

    if not args.no_clean and not args.skip_build:
        try:
            clean()
        except RuntimeError as exc:
            print(f"\n打包失败：{exc}", file=sys.stderr)
            return 1

    executable: Path | None = None
    if not args.skip_build:
        try:
            executable = build_executable(
                args.onefile, args.console, exclude_cuda=not args.with_cuda
            )
        except Exception as exc:  # noqa: BLE001 - 打包失败原因多样，统一报告
            print(f"\n打包失败：{exc}", file=sys.stderr)
            return 1

    installer: Path | None = None
    if args.installer:
        installer = build_installer(skip_build=args.skip_build)
        if installer is None:
            return 1

    print("\n打包完成：")
    report(executable, "可执行文件")
    report(installer, "安装包")

    if executable is not None or installer is not None:
        print("\n提示：模型权重与推理运行时已随包分发，安装后开箱即用；"
              "个人数据（配置/历史/日志）保存在 %LOCALAPPDATA%\\PingYiTong。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
