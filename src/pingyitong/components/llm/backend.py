"""推理后端选择。

Spark-X2.5 使用了一个专门的模型架构 ``spark2_5``，**只有 XHToken 维护的
llama.cpp 分支支持它**；上游 ``ggml-org/llama.cpp`` 不认识这个架构。
而 ``llama-cpp-python`` 捆绑的是上游 llama.cpp，因此它无法加载 Spark-X2.5
的权重——实测 0.3.35 在加载阶段直接报错::

    llama_model_load: error loading model: unknown model architecture: 'spark2_5'

这不是版本新旧的问题，而是架构支持只存在于分支里，所以换任何版本的
llama-cpp-python 都无效。

因此本项目支持两种后端，并自动选择：

======================  =========================================  ==========
后端                    适用场景                                   实现
======================  =========================================  ==========
``llama-server``        **Spark-X2.5 的推荐方式**。使用 XHToken    :mod:`.llama_server`
（外部进程）            分支预编译的可执行文件，通过本机回环地址
                        通信。无需 C++ 编译器，可选 CPU/CUDA/Vulkan。
``llama-cpp-python``    架构受上游支持的其它 GGUF 模型             :mod:`.llama_engine`
（进程内）              （Qwen、Llama 等）。Spark-X2.5 不适用。
======================  =========================================  ==========

关于"全离线"约束的说明
    ``llama-server`` 后端会在**本机回环地址（127.0.0.1）**上开一个端口，
    这是与本程序自己拉起的子进程之间的进程间通信，不涉及任何外部网络。
    程序仍然**不发起任何对外网络请求**。这一点在界面与文档中都如实说明，
    不使用"零网络"这类会被误解的表述。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ...utils import paths
from ...utils.logger import get_logger

logger = get_logger(__name__)


class BackendKind(Enum):
    """推理后端类型。"""

    #: 通过 llama-server 可执行文件（外部进程 + 本机回环通信）
    LLAMA_SERVER = "llama-server"
    #: 通过 llama-cpp-python 进程内推理
    LLAMA_CPP_PYTHON = "llama-cpp-python"
    #: 两者均不可用，进入预览模式
    NONE = "none"

    @property
    def label(self) -> str:
        """面向用户的后端名称。"""
        return {
            BackendKind.LLAMA_SERVER: "llama-server（外部进程）",
            BackendKind.LLAMA_CPP_PYTHON: "llama-cpp-python（进程内）",
            BackendKind.NONE: "不可用",
        }[self]


@dataclass
class BackendProbe:
    """后端可用性探测结果。"""

    kind: BackendKind
    #: 说明信息，可直接展示给用户
    detail: str = ""
    #: llama-server 可执行文件路径（仅 LLAMA_SERVER 有效）
    executable: str = ""
    #: 是否有可用的 CUDA 运行时（用于提示 GPU 加速状态）
    cuda_available: bool = False

    @property
    def usable(self) -> bool:
        """该后端是否可用于推理。"""
        return self.kind is not BackendKind.NONE


def discover_server_executable() -> str:
    """查找 llama-server 可执行文件。

    查找范围依次为：

    1. 环境变量 ``PINGYITONG_LLAMA_SERVER`` 指定的完整路径；
    2. ``runtime/llama/``（``scripts/download_runtime.py`` 的下载位置）；
    3. 项目根目录下的 ``llama-server.exe``；
    4. 系统 ``PATH``。

    :return: 可执行文件路径；未找到时返回空字符串。
    """
    import os
    import shutil

    override = os.environ.get("PINGYITONG_LLAMA_SERVER")
    if override:
        from pathlib import Path

        candidate = Path(override)
        if candidate.is_file():
            return str(candidate)

    for candidate in paths.llama_runtime_candidates():
        if candidate.is_file():
            return str(candidate)

    found = shutil.which("llama-server")
    return found or ""


def probe(allow_server: bool = True, allow_python: bool = True) -> BackendProbe:
    """探测可用的推理后端。

    优先级：``llama-server`` > ``llama-cpp-python``。

    之所以把外部进程后端排在前面：它是 Spark-X2.5 唯一可行的方式，
    而且不受 Python 版本与编译环境影响；同时它能用 GPU 加速。
    进程内后端仍然保留，因为对架构受上游支持的模型它更轻量
    （无需额外进程与端口）。

    :param allow_server: 是否允许探测 llama-server（用户可在设置中禁用）。
    :param allow_python: 是否允许探测 llama-cpp-python。
    """
    if allow_server:
        executable = discover_server_executable()
        if executable:
            return BackendProbe(
                kind=BackendKind.LLAMA_SERVER,
                detail=f"已找到 llama-server：{executable}",
                executable=executable,
                cuda_available=_has_cuda_runtime(executable),
            )

    if allow_python:
        from .llama_engine import llama_cpp_available

        if llama_cpp_available():
            return BackendProbe(
                kind=BackendKind.LLAMA_CPP_PYTHON,
                detail=(
                    "已安装 llama-cpp-python（仅适用于上游 llama.cpp 支持的架构；"
                    "Spark-X2.5 需要 llama-server 后端）"
                ),
            )

    if allow_server and not allow_python:
        reason = "未找到 llama-server 可执行文件"
    elif allow_python and not allow_server:
        reason = "未安装 llama-cpp-python，且未启用 llama-server 后端"
    else:
        reason = (
            "未找到 llama-server 可执行文件，也未安装 llama-cpp-python。"
            "请运行 scripts/download_runtime.py 获取推理运行时"
        )

    return BackendProbe(kind=BackendKind.NONE, detail=reason)


def _has_cuda_runtime(executable: str) -> bool:
    """检查 llama-server 同目录下是否存在 CUDA 运行时 DLL。

    llama.cpp 的 CUDA 构建需要 ``cudart64_*.dll`` 等文件。它们可能来自
    随包发布的 ``cuda-runtime`` 压缩包，也可能来自系统已安装的 CUDA。
    """
    from pathlib import Path

    directory = Path(executable).parent

    # 同目录下直接有 cudart 或 ggml-cuda 即认为 CUDA 可用
    for pattern in ("cudart64_*.dll", "ggml-cuda.dll", "cublas64_*.dll"):
        if list(directory.glob(pattern)):
            return True

    # 系统 CUDA 目录
    import os

    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        from pathlib import Path as _Path

        if list(_Path(cuda_path, "bin").glob("cudart64_*.dll")):
            return True

    return False


def has_vulkan_runtime(executable: str) -> bool:
    """检查 llama-server 同目录下是否存在 Vulkan 后端。

    Vulkan 在较新的 NVIDIA / AMD / Intel 显卡上都能用，
    对 Blackwell 等新架构是 CUDA 之外的有效加速途径。
    """
    from pathlib import Path

    directory = Path(executable).parent
    return bool(list(directory.glob("ggml-vulkan.dll")))
