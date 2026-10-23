"""静态网页抓取（联网预翻译 P3，自研）。

对应方案 v1.2 §3.7。职责边界：

- **只抓静态 HTML**：登录墙（401/403、登录表单特征）、非 200、
  robots 禁止、超限、SSRF 风险 → 明确的 Blocked/Error 结论，
  上层据此回退屏幕翻译；
- **不带登录态**：无 cookie、无 Authorization；自定义 UA 标明产品；
- **限速与截断**：单页 ≤ max_bytes（流式截断）；robots 由
  :func:`robots_allows` 判定（结果按主机缓存）；
- **正文抽取**：trafilatura（P0 选定，见 §13）；
- **下一章候选**：链接文本/rel 匹配（下一页/下一章/next/»），去重后
  交 econ_gate。

重定向：禁用 urllib 自动重定向，逐跳手动跟进并重新过 SSRF 校验。
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
import urllib.response
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from ...utils.logger import get_logger
from ...utils.urlopen_proxy_safe import open_with_proxy_fallback
from .ssrf import MAX_REDIRECTS, validate_url

logger = get_logger(__name__)

USER_AGENT = "PingYiTong/1.0 (+local desktop translator; prefetch fetcher)"

#: 登录墙启发式：正文区出现密码输入框
_LOGIN_FORM = re.compile(r"<input[^>]+type=[\"']password[\"']", re.I)

#: 下一章/下一页链接文本特征
_NEXT_TEXT = re.compile(
    r"(下一页|下一章|下一篇|next\s*page|next\s*chapter|next|»|≫|→)", re.I
)

#: robots 结果按主机缓存（host → (parser, fetched_at)）
_robots_cache: dict[str, object] = {}
_ROBOTS_TTL_S = 3600.0


@dataclass
class FetchResult:
    """抓取结论。kind: ok | blocked | error。"""

    kind: str
    reason: str = ""
    html: str = ""
    final_url: str = ""
    etag: str = ""


@dataclass
class Article:
    """抽取出的正文与翻页候选。"""

    title: str
    text_blocks: list[str] = field(default_factory=list)
    next_candidates: list[str] = field(default_factory=list)
    final_url: str = ""
    etag: str = ""


# ----------------------------------------------------------------------
# robots.txt
# ----------------------------------------------------------------------


def robots_allows(url: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    """robots.txt 判定。返回 (允许, 说明)。

    无 robots.txt 视为允许（仍受限速约束）；解析失败同允许；
    结果按主机缓存 1 小时。
    """
    import urllib.robotparser

    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    cached = _robots_cache.get(parsed.netloc)
    if cached is not None and time.perf_counter() - cached[1] < _ROBOTS_TTL_S:
        parser = cached[0]
    else:
        parser = urllib.robotparser.RobotFileParser()
        try:
            request = urllib.request.Request(robots_url, headers={"User-Agent": USER_AGENT})
            with open_with_proxy_fallback(request, timeout=timeout_s) as response:
                body = response.read(64 * 1024).decode("utf-8", "replace")
            parser.parse(body.splitlines())
        except Exception:  # noqa: BLE001 - robots 不可得按允许处理（仍限速）
            logger.debug("robots.txt 不可得（按允许处理）：%s", robots_url)
        _robots_cache[parsed.netloc] = (parser, time.perf_counter())
    allowed = parser.can_fetch(USER_AGENT, url)
    return allowed, "" if allowed else "robots.txt 禁止抓取"


# ----------------------------------------------------------------------
# 抓取（逐跳 SSRF 校验 + 流式截断 + 登录墙识别）
# ----------------------------------------------------------------------


def fetch(
    url: str,
    *,
    timeout_s: float = 15.0,
    max_bytes: int = 2 * 1024 * 1024,
    validate: bool = True,
    check_robots: bool = True,
) -> FetchResult:
    """抓取静态网页。

    :param validate: SSRF 校验开关；**生产路径必须为 True**，
        False 仅供测试在 loopback 上驱动 HTTP 行为用例。
    :return: :class:`FetchResult`；Blocked/Error 都不抛异常。
    """
    if validate:
        reason = validate_url(url)
        if reason:
            return FetchResult(kind="blocked", reason=reason)
    if check_robots:
        allowed, reason = robots_allows(url)
        if not allowed:
            return FetchResult(kind="blocked", reason=reason)

    current = url
    for _ in range(MAX_REDIRECTS + 1):
        request = urllib.request.Request(
            current, headers={"User-Agent": USER_AGENT, "Accept": "text/html"}
        )
        try:
            response = _open_noredirect(request, timeout_s)
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                # _NoRedirect 把重定向变成 HTTPError：手动跟进。
                # 重定向目标的 SSRF 校验**始终执行**，不受 validate 开关
                # 影响——validate=False 仅放宽初始 URL（测试服务器跑在
                # loopback 上），但 302 跳内网依然必须拦截。
                target = exc.headers.get("Location", "")
                if not target:
                    return FetchResult(kind="error", reason="重定向缺少 Location")
                current = urljoin(current, target)
                reason = validate_url(current)
                if reason:
                    return FetchResult(kind="blocked", reason=f"重定向被拒：{reason}")
                continue
            if exc.code in (401, 403):
                return FetchResult(kind="blocked", reason=f"HTTP {exc.code}（登录墙）")
            return FetchResult(kind="error", reason=f"HTTP {exc.code}")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return FetchResult(kind="error", reason=f"网络错误：{exc}")

        if response.status != 200:
            response.close()
            return FetchResult(kind="error", reason=f"HTTP {response.status}")

        html_bytes = response.read(max_bytes + 1)
        response.close()
        if len(html_bytes) > max_bytes:
            html_bytes = html_bytes[:max_bytes]
            logger.info("页面超过单页上限，已截断：%s", current)
        charset = response.headers.get_content_charset() or "utf-8"
        html = html_bytes.decode(charset, "replace")

        if _LOGIN_FORM.search(html):
            return FetchResult(kind="blocked", reason="页面含登录表单（疑似登录墙）")

        return FetchResult(
            kind="ok",
            html=html,
            final_url=current,
            etag=response.headers.get("ETag", ""),
        )
    return FetchResult(kind="error", reason=f"重定向超过 {MAX_REDIRECTS} 跳")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """禁用 urllib 自动重定向：逐跳手动校验。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        return None


def _open_noredirect(
    request: urllib.request.Request, timeout_s: float
) -> urllib.response.addinfourl:
    """用禁用重定向的 opener 打开请求；loopback 系统代理端点故障时降直连。

    ``build_opener`` 默认会加装跟随系统代理的 ``ProxyHandler``，因此抓取
    流量同样经过系统代理；代理客户端（fcclient 等）抖动时全套预取
    被误判 ``timeout``/``网络错误`` 而冷却。

    降级策略与判据统一由 :func:`~pingyitong.utils.urlopen_proxy_safe.
    open_with_proxy_fallback` 承担；这里只负责把 ``_NoRedirect`` 作为
    额外 handler 透传，使**首次与降级直连两条路径都保持**"手动逐跳
    重定向"语义——直连得到的 3xx 会照常变成 ``HTTPError``，交由下方
    ``fetch()`` 的 ``HTTPError(301|302|...)`` 分支手动跟进并逐跳校验。
    """
    return open_with_proxy_fallback(
        request, timeout_s, handlers=(_NoRedirect(),)
    )


# ----------------------------------------------------------------------
# 正文抽取与下一章候选
# ----------------------------------------------------------------------


def extract_article(html: str, url: str, etag: str = "") -> Article:
    """trafilatura 抽取正文（P0 选型），按空行切块。"""
    import trafilatura

    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    blocks = [block.strip() for block in (text or "").split("\n\n") if block.strip()]
    return Article(
        title=_title_of(html, url),
        text_blocks=blocks,
        next_candidates=next_candidates(html, url),
        final_url=url,
        etag=etag,
    )


def _title_of(html: str, url: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    if match:
        return " ".join(re.sub(r"<[^>]+>", "", match.group(1)).split())
    return urlparse(url).netloc


def next_candidates(html: str, base_url: str, limit: int = 5) -> list[str]:
    """下一页/下一章候选：链接文本或 rel=next 匹配，绝对化去重。"""
    candidates: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"<a\b[^>]*href=[\"']([^\"'#]+)[\"'][^>]*>(.*?)</a>", html, re.I | re.S
    ):
        href, text = match.group(1), re.sub(r"<[^>]+>", "", match.group(2))
        text = " ".join(text.split())
        if "rel=\"next\"" in match.group(0) or _NEXT_TEXT.search(text):
            absolute = urljoin(base_url, href)
            if absolute not in seen and absolute.startswith(("http://", "https://")):
                seen.add(absolute)
                candidates.append(absolute)
            if len(candidates) >= limit:
                break
    return candidates


def clear_robots_cache() -> None:
    """清空 robots 缓存（测试用）。"""
    _robots_cache.clear()
