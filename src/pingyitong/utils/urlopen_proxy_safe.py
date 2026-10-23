"""带「本机代理故障降级直连」兜底的 urlopen（自研）。

对应诊断结论（2026-10-23）：屏译通的全部 urllib 网络调用在 GUI 环境
会自动跟随 Windows 系统代理（``urllib.request.ProxyHandler`` 默认经
``getproxies()`` 读取 WinINET 注册表），fcclient 客户端开启系统代理
（本机实测 ``127.0.0.1:7892``）后流量全部经该项目中转。

问题：代理客户端抖动、切换节点或重启时，攻向代理端点的连接本身会
失败（连接被拒 / socket 超时），上层把它同"站点不可达"一律按
``网络错误``/``timeout`` 记账——日志里看到的就是连续 3 次
``timeout`` 后该站进入 COOLDOWN、预翻译停摆。而在线 AI 引擎
（``api_engine.py``）走 ``http.client`` 直连本就无视系统代理，
同一时刻两条路径行为不一致。

修复策略（最小侵入，默认保持"跟随系统代理"语义）：

1. 首次请求仍走 **默认 opener（跟随系统代理）**——尊重用户的网络
   拓扑（部分站点必须经代理才能访问），不改变任何行为。
2. 仅当首次失败**可明确归因于本机代理端点自身故障**（代理主机是
   loopback，且错误是连接被拒 / 连接超时 / read 超时）时，用
   ``ProxyHandler({})`` 空代理直连重试**一次**。
3. 系统未配置代理、代理不是本机地址（远程/用户级代理）、或错误形态
   是 DNS 失败等非代理端点故障时，**不降级**，把原始异常原样抛出——
   直连对它们没有对症意义，重试只会白付一次超时。

模块无 Qt 依赖，纯标准库。两个入口：

- :func:`open_with_proxy_fallback`——通用核心，支持透传额外 handler
  （如禁用自动重定向），供 ``web_fetcher`` 保持"手动逐跳重定向"语义；
- :func:`urlopen_with_proxy_fallback`——默认 opener 等价物，供
  ``network_monitor`` 等普通探测复用。
"""

from __future__ import annotations

import socket
import urllib.error
import urllib.request
import urllib.response
from collections.abc import Sequence
from urllib.parse import urlsplit

from .logger import get_logger

logger = get_logger(__name__)

#: 允许降级直连的代理论送本机段（代理端点自身故障只可能出现在这一段）
_LOOPBACK_HOSTS = frozenset({
    "127.0.0.1", "localhost", "::1", "[::1]",
})


def _proxy_entries() -> dict[str, str]:
    """当前系统代理映射（http/https 条目）；无代理返回空 dict。"""
    try:
        proxies = urllib.request.getproxies()
    except Exception:  # noqa: BLE001 - 注册表/环境异常都不拦住首次请求
        return {}
    # getproxies 的 no 键是绕过清单，不是端点，剥掉
    return {k: v for k, v in proxies.items() if k in ("http", "https")}


def _is_loopback(proxy_value: str) -> bool:
    """代理论送地址形如 ``http://127.0.0.1:7892``；判断其主机段。"""
    host = urlsplit(proxy_value if "//" in proxy_value else "//" + proxy_value).hostname
    if not host:
        return False
    return host.lower() in _LOOPBACK_HOSTS


def _proxy_endpoint_broke(exc: BaseException) -> bool:
    """首次失败是否可归因于 loopback 代理端点自身故障。

    只认三种形态（其余按真实站点失败处理，不降级）：
    - ``ConnectionRefusedError``（代理客户端退出了端口）
    - ``TimeoutError`` / ``socket.timeout``（连不上代理或代理无应答）
    - 包裹它们的 ``urllib.error.URLError``
    """
    inner = exc
    if isinstance(inner, urllib.error.URLError):
        inner = getattr(inner, "reason", None)
        if inner is None:
            return False
    if isinstance(inner, (ConnectionRefusedError, TimeoutError, socket.timeout)):
        return True
    # 某些环境把拒绝包成 OSError(ECONNREFUSED) 的子类链
    if isinstance(inner, OSError) and getattr(inner, "args", None):
        code = inner.args[0]
        if isinstance(code, int) and code in (
            getattr(socket, "ECONNREFUSED", 111), getattr(socket, "ETIMEDOUT", 110),
        ):
            return True
    return False


def local_proxy_endpoint_unreachable(exc: BaseException) -> bool:
    """是否应当因**本机**代理端点故障而降级直连。

    三个条件同时成立才为真（任一不成立都不降级）：

    1. 系统确实配置了 http/https 代理；
    2. 全部代理论送都是 loopback 地址（远程代理故障与本地网络无关，
       直连没有对症意义）；
    3. 异常形态属于代理端点自身故障（见 :func:`_proxy_endpoint_broke`）。

    供 :func:`open_with_proxy_fallback` 判定，也供需要自定义 opener
    （如禁用自动重定向）的调用方复用同一判据，避免两份实现漂移。
    """
    proxies = _proxy_entries()
    if not proxies:
        return False
    if not all(_is_loopback(value) for value in proxies.values()):
        return False
    return _proxy_endpoint_broke(exc)


def open_with_proxy_fallback(
    request: urllib.request.Request,
    timeout: float,
    *,
    handlers: Sequence[urllib.request.BaseHandler] = (),
) -> urllib.response.addinfourl:
    """用默认 opener 打开请求；本机代理端点故障时降级直连重试一次。

    :param request: 已带头部等信息的 :class:`urllib.request.Request`。
    :param timeout: 单次请求超时（秒）；两次尝试各用全额 ``timeout``
        （调用方的上层超时阈值本就是宽裕值，如 15s）。
    :param handlers: 额外 handler，默认与降级两条路径**同等装备**。
        传入 ``HTTPRedirectHandler`` 子类即可在降级后依旧保持自定义
        重定向语义（``web_fetcher`` 依赖这一点）。
    :return: 响应对象，语义同 :func:`urllib.request.urlopen`。
    :raises Exception: 首次/降级均失败时抛出**首次**的原始异常
        （错误文案保持与旧路径一致，不产生新的用户可见错误形态）。
    """
    try:
        if handlers:
            return urllib.request.build_opener(*handlers).open(request, timeout=timeout)
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.URLError as exc:
        # urlopen 对非 HTTP 错误包 URLError；HTTPError 继承自 URLError
        # 但站点明确回了 HTTP 状态码——那是站点的回答，不是代理故障。
        if isinstance(exc, urllib.error.HTTPError):
            raise
        first = exc
    except (ConnectionRefusedError, TimeoutError) as exc:
        first = exc

    if not local_proxy_endpoint_unreachable(first):
        raise first
    logger.warning(
        "本机系统代理端点无响应，本次请求降级直连重试：%s",
        getattr(request, "full_url", "?"),
    )
    direct_opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), *handlers
    )
    try:
        return direct_opener.open(request, timeout=timeout)
    except Exception:
        # 降级也失败：抛首次异常（代理端点故障），而不是直连的二次失败。
        # 首次异常才是"为什么降级"的根因；把二次失败抛出去会让上层
        # 把代理故障记账成站点不可达，正是本模块要消除的误导。
        raise first from None


def urlopen_with_proxy_fallback(
    request: urllib.request.Request,
    timeout: float,
) -> urllib.response.addinfourl:
    """:func:`open_with_proxy_fallback` 的无额外 handler 版本。

    与 :func:`urllib.request.urlopen` 行为一致（含 ``install_opener``
    安装的全局 opener），仅在命中降级条件时改用空代理直连重试一次。
    """
    return open_with_proxy_fallback(request, timeout)
