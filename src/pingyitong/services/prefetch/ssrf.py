"""SSRF 防护（联网预翻译 P3，自研）。

网页抓取是产品里唯一由用户输入 URL 发起网络请求的路径，必须防御
三类攻击/事故：

1. **内网探测**：`http://192.168.1.1/`、`http://127.0.0.1:8080/` 等
   ——用户被诱导输入内网地址，程序变成 SSRF 跳板；
2. **非网页协议**：`file:///etc/passwd`、`ftp://`、`data:`；
3. **重定向绕过**：公网 URL 302 跳内网——每次跳转必须重新校验。

设计：``validate_url`` 在**发起请求前**校验 scheme 与 DNS 解析结果；
重定向由 :mod:`web_fetcher` 逐跳手动跟进并重新调用本模块。
DNS 重绑定（首次解析公网、请求时解析到内网）的彻底防御需要 pin IP，
v1 以"解析结果校验 + 逐跳校验"为已知边界（§9 如实记录）。
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

#: 允许的协议——file/ftp/data 等一律拒绝
ALLOWED_SCHEMES = {"http", "https"}

#: 每跳上限（重定向逐跳校验时共用）
MAX_REDIRECTS = 5

#: 运营商级 NAT（RFC 6598）：内网语义但不在 is_private 覆盖范围内
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


def _is_dangerous_ip(ip: str) -> str | None:
    """IP 属于危险段时返回原因，安全返回 None。"""
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return "无法解析的 IP 地址"
    if address.is_loopback:
        return "loopback 地址"
    if address.is_private:
        return "私网地址"
    # CGNAT 100.64.0.0/10 是运营商内网，Python 3.11 的 is_private
    # 不覆盖该段（实测 is_private=False、is_reserved=False），
    # 不单独拦就会被当作公网地址放行。
    if address.version == 4 and address in _CGNAT_NETWORK:
        return "CGNAT 地址（100.64.0.0/10）"
    if address.is_link_local:
        return "link-local 地址"
    if address.is_reserved:
        return "保留地址"
    if address.is_multicast:
        return "组播地址"
    if address.is_unspecified:
        return "未指定地址"
    # IPv6 唯一本地地址（fc00::/7）——ipaddress 不把 ULA 算 is_private 之外的
    # 独立类别（Python 3.11 中 is_private 对 fc00::/7 返回 True，双保险再判一次）
    if address.version == 6 and address in ipaddress.ip_network("fc00::/7"):
        return "IPv6 ULA 地址"
    return None


def validate_url(url: str) -> str | None:
    """校验 URL 是否允许抓取。

    :return: ``None`` 表示允许；否则返回拒绝原因（人读）。
    """
    if not (url or "").strip():
        return "URL 为空"
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return f"协议不允许：{parsed.scheme or '(空)'}（仅 http/https）"
    if not parsed.hostname:
        return "缺少主机名"
    if parsed.username or parsed.password:
        return "不允许携带凭据"

    # DNS 解析后逐地址校验（一个域名解析到多个地址，全部必须安全）
    try:
        infos = socket.getaddrinfo(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80),
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror:
        return "域名解析失败"

    for info in infos:
        address = info[4][0]
        danger = _is_dangerous_ip(address)
        if danger:
            return f"目标解析到受限地址（{danger}：{address}）"
    return None


def validate_redirect_chain(url: str, max_hops: int = MAX_REDIRECTS) -> str | None:
    """校验重定向链中的当前一跳（web_fetcher 逐跳手动跟进时逐次调用）。

    逐跳跟进由 web_fetcher 驱动：每拿到一个重定向目标就调用本函数
    校验一次；这里只校验传入的这一个 URL，max_hops 语义由 fetcher 侧
    的跳数上限承担。
    """
    reason = validate_url(url)
    if reason:
        return f"被拒绝：{reason}"
    return None
