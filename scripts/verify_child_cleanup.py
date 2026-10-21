"""验证：父进程被强制结束时，子进程是否真的会被一起终止。

**为什么需要这个验证**

``llama-server`` 是本程序拉起的独立进程，占内存与显存都很可观。程序被
强制结束（任务管理器"结束任务"、崩溃）时没有机会执行清理代码，子进程
就变成孤儿。实测遇到过一次：攒下 8 个孤儿、合计 23.3 GB，机器内存占用
89%，显存被挤到使用共享内存，模型推理慢到超时——界面上一个字都翻不出来，
但 GPU 利用率显示 99%。

防护手段是 Windows **作业对象**（Job Object）。它的效果不能靠"代码看起来
对"来判断，必须真的强杀一次看结果，所以单独写成本脚本：

1. 起一个"父进程"，它在作业对象里拉起一个子进程（用长睡眠进程代替
   llama-server，作业对象的机制与具体程序无关）；
2. 用 ``taskkill /F`` **强杀父进程**——模拟崩溃或被任务管理器结束；
3. 检查子进程是否已经消失。

修复前：子进程仍然活着（这正是孤儿产生的方式）。
修复后：子进程随父进程一起消失。

运行：
    python scripts/verify_child_cleanup.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: 父进程脚本：建作业 -> 拉子进程 -> 把子进程 PID 打到标准输出 -> 长睡
_PARENT_SCRIPT = """
import subprocess, sys, time
sys.path.insert(0, r"{src}")
from pingyitong.utils.process_guard import create_kill_on_close_job, assign_to_job

job = create_kill_on_close_job()
if job is None:
    print("作业对象创建失败", flush=True)
    raise SystemExit(3)

# 子进程：长睡眠，模拟常驻的推理后端
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
assign_to_job(child.pid)
print(child.pid, flush=True)
time.sleep(600)
"""


def _is_alive(pid: int) -> bool:
    """检查进程是否仍然存在。"""
    import psutil

    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except Exception:  # noqa: BLE001
        return False


def _verify(parent: subprocess.Popen[str], child_pids: list[int]) -> int:
    # 等父进程把子进程 PID 打出来
    child_pid: int | None = None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        line = parent.stdout.readline() if parent.stdout else ""
        line = (line or "").strip()
        if line.isdigit():
            child_pid = int(line)
            child_pids.append(child_pid)
            break
        if parent.poll() is not None:
            break

    if child_pid is None:
        if parent.poll() is None:
            parent.kill()
        _, stderr = parent.communicate(timeout=10)
        print("!! 未能取得子进程 PID；父进程输出异常")
        print("   stderr:", stderr[:400])
        return 2

    print(f"父进程 PID : {parent.pid}")
    print(f"子进程 PID : {child_pid}")
    time.sleep(1.0)

    if not _is_alive(child_pid):
        print("!! 子进程在父进程存活时就不见了，验证无法进行")
        return 2

    print("子进程存活 OK（父进程仍在运行）")
    print()
    print(f"强制结束父进程（taskkill /F /PID {parent.pid}）…")
    subprocess.run(
        ["taskkill", "/F", "/PID", str(parent.pid)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    parent.wait(timeout=10)

    # 内核回收作业对象需要一点时间
    for waited in range(20):
        time.sleep(0.25)
        if not _is_alive(child_pid):
            print(f"子进程已消失 OK（父进程结束后 {waited * 0.25:.2f}s 内）")
            print()
            print("结论：作业对象生效，强制结束程序不会留下孤儿进程。")
            return 0

    print(f"!! 子进程 {child_pid} 在父进程被强杀后仍然存活 —— 防护未生效")
    return 1


def main() -> int:
    print("=" * 68)
    print("验证：强杀父进程后，子进程是否会被一并终止")
    print("=" * 68)

    parent = subprocess.Popen(
        [sys.executable, "-c", _PARENT_SCRIPT.format(src=str(ROOT / "src"))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    child_pids: list[int] = []
    try:
        return _verify(parent, child_pids)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)
        for child_pid in child_pids:
            if _is_alive(child_pid):
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(child_pid)],
                    capture_output=True,
                    check=False,
                )


if __name__ == "__main__":
    raise SystemExit(main())
