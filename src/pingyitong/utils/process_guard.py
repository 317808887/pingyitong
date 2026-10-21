"""子进程生命周期守卫。

**要解决的问题**

推理后端 ``llama-server`` 是本程序拉起的独立进程（见
``docs/架构说明.md`` 二·2.11）。它占用的资源相当可观：两个模型同时在位时
每个进程约 1–6 GB 内存，还要在显存里分配权重与计算缓冲区。

正常退出时程序会逐个终止它们，但如果程序是**被强制结束**的——任务管理器
"结束任务"、崩溃、断电——清理代码根本没有机会执行，这些子进程就变成了
孤儿：它们继续占着内存和显存，而用户看不到任何窗口，只能看到"内存快满了、
GPU 跑满却什么都不干"。

实测遇到过一次：反复强制结束程序后攒下 **8 个孤儿进程、合计 23.3 GB 内存**，
机器内存占用 89%，GPU 显存被挤到使用共享内存（走 PCIe），于是模型推理
慢到超时，界面上一个字都翻不出来——但 GPU 利用率显示 99%，看上去"在忙"。

**两层防护**

1. :func:`create_kill_on_close_job` / :func:`assign_to_job`：
   用 Windows **作业对象**（Job Object）把子进程绑到父进程上。
   父进程无论是正常退出、崩溃还是被强杀，内核关闭其句柄后作业对象随之销毁，
   其中的进程会被**内核直接终止**——不依赖我们写任何清理代码。
2. :func:`reap_stale_servers`：
   启动时清理**上一次**遗留且有应用归属记录的孤儿进程。旧版本未记录的
   进程保守保留，避免影响仍在运行的实例或用户手动启动的服务。

两者都只在 Windows 上生效，其他平台安全降级为空操作。
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import math
import os
import sys
import uuid
from ctypes import wintypes
from pathlib import Path

from . import paths
from .logger import get_logger

logger = get_logger(__name__)

#: 推理后端进程的可执行文件名
SERVER_IMAGE_NAME = "llama-server.exe"

# --------------------------------------------------------------------------
# 作业对象（Job Object）
# --------------------------------------------------------------------------

#: 作业关闭时终止其中所有进程
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

#: 作业信息类别：扩展限制信息
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

#: OpenProcess 需要的权限：设置配额 + 终止
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001

#: 进程句柄，作业对象在程序运行期间一直持有，不能让它被回收
_job_handle: int | None = None
_job_failed = False


class _IO_COUNTERS(ctypes.Structure):  # noqa: N801 - Win32 结构体保持 ABI 原名
    """Win32 ``IO_COUNTERS``。"""

    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):  # noqa: N801 - Win32 ABI 原名
    """Win32 ``JOBOBJECT_BASIC_LIMIT_INFORMATION``。"""

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):  # noqa: N801 - Win32 ABI 原名
    """Win32 ``JOBOBJECT_EXTENDED_LIMIT_INFORMATION``。"""

    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _kernel32():
    """声明 Win32 ABI，避免 ctypes 默认的 32 位 int 截断 64 位句柄。"""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def create_kill_on_close_job() -> int | None:
    """创建一个"关闭即终止"的作业对象。

    :return: 作业句柄；平台不支持或创建失败时返回 ``None``。
    """
    global _job_handle, _job_failed

    if sys.platform != "win32":
        return None
    if _job_handle is not None:
        return _job_handle
    if _job_failed:
        return None

    handle = None
    try:
        kernel32 = _kernel32()
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW 失败")

        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject 失败")

        _job_handle = handle
        logger.debug("已创建作业对象：父进程退出时自动终止子进程")
        return handle
    except Exception as exc:  # noqa: BLE001 - 作业对象不可用时不应影响推理
        if handle:
            with contextlib.suppress(Exception):
                kernel32.CloseHandle(handle)
        _job_failed = True
        logger.warning("无法创建作业对象，子进程将不会随程序退出而终止：%s", exc)
        return None


def assign_to_job(process_id: int) -> bool:
    """把进程加入"关闭即终止"的作业。

    :param process_id: 目标进程 ID。
    :return: 是否成功加入。失败不影响程序运行，只是退化为没有这层保护。
    """
    # Record only an actual child of this process.  This also covers a failed
    # Job Object assignment; runtime directory membership alone is not ownership.
    _record_owned_server(process_id)
    job = create_kill_on_close_job()
    if job is None:
        return False

    try:
        kernel32 = _kernel32()
        handle = kernel32.OpenProcess(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, int(process_id)
        )
        if not handle:
            raise OSError(ctypes.get_last_error(), "OpenProcess 失败")
        try:
            if not kernel32.AssignProcessToJobObject(job, handle):
                raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject 失败")
        finally:
            kernel32.CloseHandle(handle)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("将进程 %s 加入作业失败（可忽略）：%s", process_id, exc)
        return False


# --------------------------------------------------------------------------
# 清理遗留的孤儿进程
# --------------------------------------------------------------------------


def _ownership_dir() -> Path:
    """应用私有的推理子进程身份记录；不包含模型输入或密钥。"""
    return paths.app_data_dir() / "runtime" / "server_ownership"


def _record_owned_server(process_id: int) -> None:
    """记录本进程实际创建的 llama 子进程，记录失败不影响推理。"""
    if sys.platform != "win32":
        return
    temporary: Path | None = None
    try:
        import psutil

        child = psutil.Process(int(process_id))
        owner = child.parent()
        if owner is None or owner.pid != os.getpid() or not owner.is_running():
            return
        executable = Path(child.exe()).resolve()
        if executable.name.lower() != SERVER_IMAGE_NAME:
            return
        child_created = child.create_time()
        owner_created = owner.create_time()
        if not all(
            math.isfinite(value) and value > 0 for value in (child_created, owner_created)
        ):
            return
        if owner_created > child_created:
            return
        record = {
            "schema": 1,
            "pid": child.pid,
            "created": child_created,
            "executable": str(executable),
            "owner_pid": owner.pid,
            "owner_created": owner_created,
        }
        directory = _ownership_dir()
        directory.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256(
            f"{child.pid}:{child_created.hex()}".encode("ascii")
        ).hexdigest()[:16]
        destination = directory / f"{child.pid}-{identity}.json"
        temporary = directory / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        temporary.replace(destination)
    except Exception:  # noqa: BLE001 - unavailable identity means do not reap it
        logger.debug("无法记录推理子进程身份，启动清理会保守跳过它", exc_info=True)
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)


def _read_ownership_record(path: Path) -> dict | None:
    """只接受完整身份；损坏、未知版本或权限不足均不能授权终止进程。"""
    try:
        if path.stat().st_size > 4096:
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(record, dict)
            or type(record.get("schema")) is not int
            or record["schema"] != 1
        ):
            return None
        for key in ("pid", "owner_pid"):
            if type(record.get(key)) is not int or record[key] <= 0:
                return None
        for key in ("created", "owner_created"):
            value = record.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                return None
        if record["owner_created"] > record["created"]:
            return None
        if (
            not isinstance(record.get("executable"), str)
            or not Path(record["executable"]).is_absolute()
        ):
            return None
        return record
    except (OSError, ValueError):
        return None


def _owner_alive(record: dict, psutil) -> bool:
    """只有确证原启动者消失才返回 False；PID 复用由创建时刻区分。"""
    try:
        owner = psutil.Process(record["owner_pid"])
        if owner.create_time() != record["owner_created"]:
            return False
        return bool(owner.is_running())
    except psutil.NoSuchProcess:
        return False
    # AccessDenied and other ambiguous identity errors propagate to the caller,
    # which conservatively skips this record.


def reap_stale_servers(executable: str | Path) -> list[int]:
    """终止上次运行遗留的推理后端进程。

    仅回收带完整应用归属记录、PID/创建时刻/完整 exe 均一致且原启动者
    确认已消失的进程。仍有启动者、用户手动运行、权限或身份不明的服务
    均保留。旧版本未记录归属的进程不能只因处于同一运行时目录而被终止。

    :param executable: 本程序使用的 ``llama-server`` 可执行文件路径。
    :return: 被终止的进程 ID 列表。
    """
    if sys.platform != "win32":
        return []

    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil 为硬依赖
        logger.debug("未安装 psutil，跳过孤儿进程清理")
        return []

    try:
        target = Path(executable).resolve(strict=True)
        if not target.is_file() or target.name.lower() != SERVER_IMAGE_NAME:
            return []
        directory = _ownership_dir()
        records = list(directory.glob("*.json"))
    except OSError:
        return []

    killed: list[int] = []
    for record_path in records:
        try:
            record = _read_ownership_record(record_path)
            if record is None or Path(record["executable"]).resolve() != target:
                continue
            process = psutil.Process(record["pid"])
            if process.create_time() != record["created"]:
                # A PID reused by another process must never be terminated.
                record_path.unlink(missing_ok=True)
                continue
            if (
                Path(process.exe()).resolve() != target
                or process.name().lower() != SERVER_IMAGE_NAME
            ):
                continue
            if process.ppid() != record["owner_pid"] or _owner_alive(record, psutil):
                continue
            # Defend against a live parent appearing/reused PID between reads.
            # psutil.parent() itself compares parent and child creation times.
            if process.parent() is not None:
                continue
            # create_time() is cached by psutil; is_running() refreshes PID identity.
            if not process.is_running():
                record_path.unlink(missing_ok=True)
                continue
            if _owner_alive(record, psutil):
                continue
            process.kill()
            killed.append(process.pid)
            record_path.unlink(missing_ok=True)
        except psutil.NoSuchProcess:
            # The recorded child exited (or its PID was reused); it no longer
            # needs an ownership record. Unknown permissions retain the record.
            with contextlib.suppress(OSError):
                record_path.unlink(missing_ok=True)
        except psutil.AccessDenied:
            continue
        except Exception:  # noqa: BLE001 - 清理失败不应影响启动
            logger.debug("清理孤儿进程时出错（已跳过）", exc_info=True)

    if killed:
        logger.info(
            "已清理上一次运行遗留的推理进程 %s 个（PID %s）",
            len(killed),
            ", ".join(str(pid) for pid in killed),
        )
    return killed


def reset_for_tests() -> None:
    """重置内部状态，仅供测试使用。"""
    global _job_handle, _job_failed
    _job_handle = None
    _job_failed = False
