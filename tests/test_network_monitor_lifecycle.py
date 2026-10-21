"""NetworkMonitor lifecycle contract: idempotent start, real stop."""
from __future__ import annotations

import threading
import time

from pingyitong.services.prefetch.network_monitor import NetworkMonitor


def _alive_probes() -> int:
    return sum(1 for t in threading.enumerate() if t.name == "net-probe")


def test_start_is_idempotent_and_stop_releases_thread():
    monitor = NetworkMonitor(probe_interval_s=0.2, timeout_ms=200)
    before = _alive_probes()
    monitor.start()
    monitor.start()
    monitor.start()
    time.sleep(0.5)
    # 多次 start 只允许一个存活探测线程
    assert _alive_probes() - before <= 1, "重复 start 不得累积探测线程"
    monitor.stop()
    deadline = time.time() + 3
    while time.time() < deadline and _alive_probes() > before:
        time.sleep(0.1)
    assert _alive_probes() == before, "stop 后探测线程应退出"
    # stop 幂等
    monitor.stop()
    monitor.stop()


def test_stop_wakes_sleeping_loop_promptly():
    monitor = NetworkMonitor(probe_interval_s=30, timeout_ms=200)
    monitor.start()
    started = time.monotonic()
    monitor.stop()
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"stop 应立刻唤醒 30s 睡眠循环，实际 {elapsed:.2f}s"
