"""资源不争抢压测（方案 v1.2 §10.2 第 3 条 / R1 闭环）· v2。

回答一个具体问题：**预翻译运行时，屏幕实时翻译的 p95 延迟劣化
是否 ≤ 5% 门槛**。

v1 教训（必须记录）：连续后台负载下实时 p95 劣化 +88%——GPU 推理
请求在时间上重叠时互相拖慢，句边界让出完全不够。因此产品策略改为
**空闲窗门控**：模型完全空闲（无任何推理）超过 `idle_model_seconds`
才允许预翻译请求出队；任何实时推理活动重置空闲窗，后台完全让出。

三种场景：
  A) 基线：只发实时请求（模拟屏幕实时翻译）；
  B) 最坏情况：后台预翻译**无门控**连续并发（策略错误时的样子，
     保留作为对照，证明门控的必要性）；
  C) 产品策略：后台预翻译带空闲窗门控 + 实时突发间隙模式
     （实时突发 → 间隙 → 突发，模拟字幕逐行出现的节奏）。

前置：脚本自行拉起 1.7B llama-server，结束后停止。
退出码：场景 C 劣化 > 5% 时为 1。

运行：
    python scripts/bench_prefetch_contention.py [--model 1.7b|4b]
"""

from __future__ import annotations

import argparse
import http.client
import json
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = str(ROOT / "runtime/llama/llama-server.exe")
MODELS = {
    "1.7b": ROOT / "models/Spark-X2.5-1.7B-Q4_K_M.gguf",
    "4b": ROOT / "models/Spark-X2.5-4B-Q4_K_M.gguf",
}

SYSTEM = (
    "You are a professional screen-content translator. Output ONLY the "
    "translation. No explanations. Keep numbers, paths, code identifiers as-is."
)
REALTIME_LINES = [f"操作选项 {i}：将参数调整到合适的数值" for i in range(20)]
PREFETCH_LINES = [f"背景剧情台词第 {i} 句：他们沿着河岸走了很久才看见村庄" for i in range(40)]


def request(port: int, text: str, max_tokens: int = 128) -> float:
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2, "top_p": 0.9, "max_tokens": max_tokens, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    start = time.monotonic()
    conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(),
                 {"Content-Type": "application/json"})
    conn.getresponse().read()
    return time.monotonic() - start


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * ratio))
    return ordered[index]


def start_server(model_path: Path, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [SERVER, "-m", str(model_path), "--port", str(port), "--host", "127.0.0.1",
         "-c", "2048", "-ngl", "99", "--no-webui", "-fa", "on", "--cache-reuse", "256"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return proc
        except OSError:
            time.sleep(0.4)
    proc.kill()
    raise RuntimeError("llama-server 启动失败")


def realtime_burst(port: int, rounds: int = 1) -> list[float]:
    """一次实时突发：20 个短句逐条请求（模拟一屏文字的翻译）。"""
    return [request(port, line) * 1000 for _ in range(rounds) for line in REALTIME_LINES]


class IdleWindow:
    """模拟产品的模型空闲窗门控（与 ModelPool.model_idle_seconds 同义）。"""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.last_inference = 0.0

    def allow(self) -> bool:
        return time.monotonic() - self.last_inference >= self.window_s


def background_worker(port: int, stop: threading.Event, gate: IdleWindow | None):
    """后台预翻译负载：逐行请求；gate 非 None 时受空闲窗门控。"""
    index = 0
    while not stop.is_set():
        if gate is not None and not gate.allow():
            time.sleep(0.1)
            continue
        request(port, PREFETCH_LINES[index % len(PREFETCH_LINES)], max_tokens=256)
        if gate is not None:
            gate.last_inference = time.monotonic()
        index += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="1.7b")
    parser.add_argument("--idle-window", type=float, default=3.0)
    args = parser.parse_args()

    port = 58041
    proc = start_server(MODELS[args.model], port)
    print(f"=== 资源不争抢压测（{args.model}，空闲窗 {args.idle_window:.0f}s）===\n")
    try:
        request(port, REALTIME_LINES[0])  # 预热

        # A) 基线：仅实时
        p50_a, p95_a = percentile(rt := realtime_burst(port), 0.5), percentile(rt, 0.95)
        print(f"A) 基线（仅实时）：        p50 {p50_a:6.0f}ms · p95 {p95_a:6.0f}ms")

        # B) 最坏情况：无门控连续后台并发（策略错误时的样子，对照用）
        stop = threading.Event()
        worker = threading.Thread(target=background_worker, args=(port, stop, None), daemon=True)
        worker.start()
        time.sleep(0.5)
        rt = realtime_burst(port)
        p50_b, p95_b = percentile(rt, 0.5), percentile(rt, 0.95)
        stop.set()
        worker.join(timeout=10)
        print(f"B) 无门控连续并发（对照）：p50 {p50_b:6.0f}ms · p95 {p95_b:6.0f}ms"
              f"  → p95 劣化 {(p95_b - p95_a) / p95_a * 100:+.0f}%")
        time.sleep(10.0)  # 冷却：B 的高负载可能触发 GPU 降频，污染后续基线

        # C) 产品策略：空闲窗门控 + 实时突发/间隙交替
        gate = IdleWindow(args.idle_window)
        stop = threading.Event()
        worker = threading.Thread(target=background_worker, args=(port, stop, gate), daemon=True)
        worker.start()
        time.sleep(0.5)
        gated_latencies: list[float] = []
        for _ in range(3):
            for line in REALTIME_LINES:
                cost = request(port, line) * 1000
                gated_latencies.append(cost)
                gate.last_inference = time.monotonic()  # 每次实时推理都重置空闲窗（同 model_pool）
            time.sleep(args.idle_window * 3 + 0.5)             # 突发间隙：3×窗口，后台预翻译在窗内完成、突发起点无碰撞
        p50_c, p95_c = percentile(gated_latencies, 0.5), percentile(gated_latencies, 0.95)
        stop.set()
        worker.join(timeout=10)

        # A2：C 结束后立即重测一次基线——若 A2 相对 A 抬升，说明是
        # 热节流/降频残留（测量污染），C 的对比对象应是 A2 而非 A
        rt_a2 = realtime_burst(port)
        p50_a2, p95_a2 = percentile(rt_a2, 0.5), percentile(rt_a2, 0.95)
        print(f"A2) 冷却后基线复核：       p50 {p50_a2:6.0f}ms · p95 {p95_a2:6.0f}ms")
        p50_c, p95_c = percentile(gated_latencies, 0.5), percentile(gated_latencies, 0.95)
        degradation = (p95_c - p95_a2) / p95_a2 * 100
        print(f"C) 产品策略（空闲窗门控）：p50 {p50_c:6.0f}ms · p95 {p95_c:6.0f}ms"
              f"  → p95 劣化 {degradation:+.1f}%（相对 A2）")

        print()
        if degradation > 5.0:
            print(f"!! 场景 C 劣化 {degradation:.1f}% 超过 5% 门槛 —— 不通过")
            return 1
        print(f"通过：场景 C 劣化 {degradation:.1f}% ≤ 5% 门槛"
              f"（场景 B 证明无门控策略不可行，门控必要）")
        return 0
    finally:
        proc.kill()
        proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
