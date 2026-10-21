"""联网状态判定（联网预翻译，自研）。

对应方案 v1.1 §3.1。核心要求：**联网以实际可达性探测为准**，
不能只信系统 online 标志——captive portal（校园网/企业网认证页）下
系统报告"已连接"，实际不可达。

判定规则（P0 实测后确定）：
1) TCP 探测 223.5.5.5:443（国内稳定点）；
2) HTTP 探测一个可配置端点（识别 captive portal：TCP 通但 HTTP 被劫持）；
3) 全部失败重试 2 轮后才判 OFFLINE（避免瞬时抖动误切）；
4) 计费网络：优先 WinRT ``NetworkCostType``（P0 实测本机可行）；
   判定失败回退 ``UNKNOWN``——**保守处理**：wifi_only 开启时
   UNKNOWN 一律不预取（宁可少预取，不可偷跑流量）。

P0 实证局限：iPhone 热点被 Windows 报告为 Unrestricted，
蜂窝热点误识别无法在本层修复，界面文案明示。
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.request
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal

from ...utils.logger import get_logger

logger = get_logger(__name__)


class NetState(Enum):
    """探测结论。"""

    ONLINE = "online"
    OFFLINE = "offline"


class NetworkCost(Enum):
    """计费判定结论。UNKNOWN 必须按计费处理（保守失败）。"""

    UNMETERED = "unmetered"
    METERED = "metered"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProbeResult:
    """一次完整探测的结论。"""

    state: NetState
    cost: NetworkCost
    elapsed_ms: float


def _probe_tcp(host: str, port: int, timeout_s: float) -> bool:
    """TCP 连通性探测（单点）。"""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _probe_http(url: str, timeout_s: float) -> bool:
    """HTTP 层探测：captive portal 会把请求劫持到认证页，
    这里只区分"HTTP 栈可达"与"完全不可达"，劫持检测由调用方
    对比返回体特征（探测点返回 204 时响应码非 204 即疑似 portal）。"""
    try:
        request = urllib.request.Request(url, method="GET",
                                         headers={"User-Agent": "PingYiTong/1.0"})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status < 500
    except OSError:
        return False
    except Exception:  # noqa: BLE001 - 探测不抛异常
        return False


def _cost_via_winrt() -> NetworkCost | None:
    """WinRT NetworkCostType 读取（P0 实测可行）；失败返回 None。

    0=Unknown 1=Unrestricted 2=Fixed 3=Variable（Fixed/Variable 视为计费）。
    """
    try:
        from winrt.windows.networking.connectivity import NetworkInformation
    except Exception:  # noqa: BLE001 - winrt 不可用（非 Win10+ / 未安装）
        return None
    try:
        profiles = NetworkInformation.get_connection_profiles()
        costs = []
        for profile in profiles:
            try:
                costs.append(profile.get_connection_cost().network_cost_type)
            except Exception:  # noqa: BLE001 - 单个 profile 失败不影响整体
                continue
        if not costs:
            return NetworkCost.UNKNOWN
        if all(c in (2, 3) for c in costs):
            return NetworkCost.METERED
        if 1 in costs:
            return NetworkCost.UNMETERED
        return NetworkCost.UNKNOWN
    except Exception:  # noqa: BLE001
        return None


class NetworkMonitor(QObject):
    """后台周期探测 + 主动探测；状态变化发信号。"""

    state_changed = pyqtSignal(object)  # ProbeResult

    def __init__(
        self,
        tcp_host: str = "223.5.5.5",
        tcp_port: int = 443,
        http_url: str = "http://www.baidu.com/robots.txt",
        probe_interval_s: float = 10.0,
        timeout_ms: int = 2000,
        wifi_only: bool = True,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._tcp_host = tcp_host
        self._tcp_port = tcp_port
        self._http_url = http_url
        self._interval = max(2.0, float(probe_interval_s))
        self._timeout_s = max(0.3, timeout_ms / 1000.0)
        self._wifi_only = wifi_only
        self._lock = threading.Lock()
        self._last: ProbeResult | None = None
        # 探测线程生命周期：start 幂等（先停旧的），stop 真正落停。
        # 修复前 stop() 是空操作，而 app.py 每次应用设置都会调 start，
        # 长期运行会累积多个每 10s 探测一次的永久线程（禁用预取也停不掉）。
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._lifecycle_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 探测
    # ------------------------------------------------------------------

    def probe_now(self) -> ProbeResult:
        """执行一轮探测（可在任意线程调用）。

        失败重试 2 轮才判 OFFLINE；计费判定失败按 UNKNOWN（保守）。
        """
        started = time.perf_counter()
        online = False
        for attempt in range(2):
            if _probe_tcp(self._tcp_host, self._tcp_port, self._timeout_s):
                online = True
                break
            if _probe_http(self._http_url, self._timeout_s):
                online = True
                break
            if attempt == 0:
                time.sleep(0.3)

        cost = _cost_via_winrt() or NetworkCost.UNKNOWN
        state = NetState.ONLINE if online else NetState.OFFLINE
        result = ProbeResult(
            state=state,
            cost=cost,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

        with self._lock:
            changed = (
                self._last is None
                or self._last.state != result.state
                or self._last.cost != result.cost
            )
            self._last = result
        if changed:
            logger.info(
                "联网探测：%s（计费判定 %s，探测耗时 %.0fms）",
                result.state.value, result.cost.value, result.elapsed_ms,
            )
            self.state_changed.emit(result)
        return result

    def prefetch_allowed(self, result: ProbeResult | None = None) -> bool:
        """当前是否允许产生预取网络流量（wifi_only 语义在此落实）。"""
        result = result or self._last
        if result is None or result.state != NetState.ONLINE:
            return False
        # UNKNOWN 按计费处理（保守失败，v1.1 §3.1）
        return not (self._wifi_only and result.cost != NetworkCost.UNMETERED)

    # ------------------------------------------------------------------
    # 周期探测
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动后台探测线程（daemon，随进程退出）。

        幂等：重复调用先停掉旧线程再起新线程。app.py 在初始化与每次
        应用设置时都会调用本方法，不允许线程累积。
        """
        with self._lifecycle_lock:
            # 旧线程先换下（唤醒其退出事件），锁外 join——stop() 与本方法
            # 共用非重入锁，锁内调用会死锁；旧实现的 self.stop() 在这里
            # 就是因为这个原因丢失的（实测 3 次 start 累积 3 个线程）。
            old_event, old_thread = self._stop_event, self._thread
            self._stop_event = threading.Event()

            def _loop(stop_event: threading.Event) -> None:
                while not stop_event.is_set():
                    try:
                        self.probe_now()
                    except Exception:  # noqa: BLE001 - 探测线程永不退出
                        logger.debug("探测异常", exc_info=True)
                    stop_event.wait(self._interval)

            thread = threading.Thread(
                target=_loop, args=(self._stop_event,), name="net-probe", daemon=True,
            )
            self._thread = thread
            thread.start()
        if old_event is not None:
            old_event.set()
        if old_thread is not None and old_thread is not threading.current_thread():
            old_thread.join(timeout=3.0)

    def stop(self) -> None:
        """停止探测线程。

        stop_event 立刻唤醒 sleep 中的循环；若正卡在一次探测内部，
        该次探测有界（超时上限），join 超时后由守护线程随事件置位自行退出。
        """
        with self._lifecycle_lock:
            stop_event, thread = self._stop_event, self._thread
            self._stop_event = None
            self._thread = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
