"""代理故障降级直连的单元/集成测试（配合 utils/urlopen_proxy_safe.py）。

背景：Windows GUI 环境下 urllib 会跟随系统代理（fcclient 本机
loopback 代理 127.0.0.1:7892）。代理客户端抖动时，屏译通的整条
urllib 网络路径（抓取/robots/HTTP 探测）会把「代理端点自身故障」
误判为「站点超时」，预翻译源被连续 3 次 timeout 误冷却。

测试策略：
- 集成用例：起真实 loopback http.server 当目标站点，把系统代理
  ``getproxies()`` mock 到一个必然立即 ECONNREFUSED 的 loopback
  死端口（127.0.0.1:1）——首次打开真实失败，断言降级直连后成功；
- 单元用例：mock urlopen / build_opener，断言降级触发条件（未配置
  代理不降、远程代理不降、HTTP 站点回复不降、DNS 错误不降）。
"""

from __future__ import annotations

import http.server
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

import pingyitong.utils.urlopen_proxy_safe as ups
from pingyitong.services.prefetch.web_fetcher import clear_robots_cache, fetch
from pingyitong.utils.urlopen_proxy_safe import (
    _proxy_endpoint_broke,
    urlopen_with_proxy_fallback,
)

#: 必然无人监听的 loopback 端口（连接立即被拒，不等待超时）
_DEAD_PROXY_PORT = 1
_DEAD_PROXY_URL = f"http://127.0.0.1:{_DEAD_PROXY_PORT}"


class _Handler(http.server.BaseHTTPRequestHandler):
    """本地测试站点：/ /redirect /target /robots.txt"""

    def log_message(self, *args):  # 静默
        pass

    def do_GET(self) -> None:
        if self.path == "/robots.txt":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"User-agent: *\nAllow: /\n")
            return
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/target")
            self.end_headers()
            return
        body = {
            "/": "<html><body><p>pingyi-e2e-ok</p></body></html>",
            "/target": "<html><body><p>redirect-target-ok</p></body></html>",
        }.get(self.path, "<html><body><p>not-found</p></body></html>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode())


class TestProxyFallbackE2E(unittest.TestCase):
    """真实 TCP 失败注入：死代理端口 → 降级直连 → 成功抓取。"""

    server = None  # type: ignore[var-annotated]
    host = ""
    port = 0

    @classmethod
    def setUpClass(cls) -> None:
        clear_robots_cache()
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        cls.host = f"http://127.0.0.1:{cls.port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        assert cls.server is not None
        cls.server.shutdown()
        clear_robots_cache()

    def _with_dead_proxy(self):
        proxies = {"http": _DEAD_PROXY_URL, "https": _DEAD_PROXY_URL}
        return mock.patch("urllib.request.getproxies", return_value=proxies)

    def test_fetch_falls_back_to_direct_on_dead_loopback_proxy(self) -> None:
        with self._with_dead_proxy():
            result = fetch(f"{self.host}/", validate=False, check_robots=False)
        self.assertEqual("ok", result.kind, f"降级直连后应成功：{result.reason}")
        self.assertIn("pingyi-e2e-ok", result.html)

    def test_redirect_semantics_kept_after_fallback(self) -> None:
        """降级直连仍保持「手动逐跳重定向」语义。

        重定向目标校验在 fetch 内恒为开启（对 loopback 目标按设计拒绝），
        这里 mock 掉 validate_url 以专注验证传输层行为；
        SSRF 拦截本身由 test_prefetch_p3.py 的 TestSSRF 覆盖。
        """
        with self._with_dead_proxy(), \
                mock.patch(
                    "pingyitong.services.prefetch.web_fetcher.validate_url",
                    return_value=None,
                ):
            result = fetch(f"{self.host}/redirect", validate=False, check_robots=False)
        self.assertEqual("ok", result.kind, f"重定向应手动跟进成功：{result.reason}")
        self.assertIn("redirect-target-ok", result.html)

    def test_robots_fetch_survives_dead_proxy(self) -> None:
        with self._with_dead_proxy():
            result = fetch(f"{self.host}/", validate=False, check_robots=True)
        self.assertEqual("ok", result.kind, f"robots 路径不应阻断降级抓取：{result.reason}")


class TestProxyFallbackConditions(unittest.TestCase):
    """降级触发条件：何时降、何时果断不降。"""

    DEAD = {"http": _DEAD_PROXY_URL, "https": _DEAD_PROXY_URL}
    REMOTE = {"http": "http://192.0.2.1:8080", "https": "http://192.0.2.1:8080"}

    @staticmethod
    def _request() -> urllib.request.Request:
        return urllib.request.Request("http://example.com/ok")

    def test_no_proxy_configured_no_fallback(self) -> None:
        # 真实 urlopen 会把 WinError 10061 包成 URLError：
        # 语义只要求「原样抛出、不进入降级」，不对包装层做额外翻译。
        with (mock.patch("urllib.request.getproxies", return_value={}),
              mock.patch("urllib.request.build_opener") as build):
            self.assertRaises(
                urllib.error.URLError,
                urlopen_with_proxy_fallback, self._request(), 5.0,
            )
        build.assert_not_called()

    def test_remote_proxy_failure_no_fallback(self) -> None:
        with (mock.patch("urllib.request.getproxies", return_value=self.REMOTE),
              mock.patch("urllib.request.build_opener") as build):
            self.assertRaises(
                urllib.error.URLError,
                urlopen_with_proxy_fallback, self._request(), 5.0,
            )
        build.assert_not_called()

    def test_http_site_answer_not_fallback(self) -> None:
        """站点明确回了 HTTP 状态码——那是站点的回答，不降级重试。"""
        err = urllib.error.HTTPError(
            "http://example.com/404", 404, "Not Found", {}, mock.Mock())
        with (mock.patch("urllib.request.getproxies", return_value=self.DEAD),
              mock.patch("urllib.request.urlopen", side_effect=err),
              mock.patch("urllib.request.build_opener") as build):
                    self.assertRaises(
                        urllib.error.HTTPError,
                        urlopen_with_proxy_fallback, self._request(), 5.0,
                    )
        build.assert_not_called()

    def test_dns_error_not_fallback(self) -> None:
        baton = urllib.error.URLError(OSError(101, "getaddrinfo failed"))
        with (mock.patch("urllib.request.getproxies", return_value=self.DEAD),
              mock.patch("urllib.request.urlopen", side_effect=baton),
              mock.patch("urllib.request.build_opener") as build):
                    self.assertRaises(
                        urllib.error.URLError,
                        urlopen_with_proxy_fallback, self._request(), 5.0,
                    )
        build.assert_not_called()

    def test_loopback_timeout_falls_back(self) -> None:
        urlexc = urllib.error.URLError(TimeoutError("timed out"))
        sentinel = mock.sentinel.response
        with (mock.patch("urllib.request.getproxies", return_value=self.DEAD),
              mock.patch("urllib.request.urlopen", side_effect=urlexc),
              mock.patch("urllib.request.build_opener") as build):
                    build.return_value.open.return_value = sentinel
                    result = urlopen_with_proxy_fallback(self._request(), 5.0)
        self.assertIs(sentinel, result)


class TestEndpointBrokenClassifier(unittest.TestCase):
    """_proxy_endpoint_broke 的错误形态白名单。"""

    def test_refused_and_timeout_true(self) -> None:
        self.assertTrue(_proxy_endpoint_broke(ConnectionRefusedError("refused")))
        self.assertTrue(_proxy_endpoint_broke(TimeoutError("timed out")))

    def test_bare_socket_timeout_true(self) -> None:
        self.assertTrue(_proxy_endpoint_broke(TimeoutError("timed out")))

    def test_urlerror_wrapped_refused_true(self) -> None:
        self.assertTrue(_proxy_endpoint_broke(
            urllib.error.URLError(ConnectionRefusedError("refused"))))

    def test_urlerror_wrapped_timeout_true(self) -> None:
        self.assertTrue(_proxy_endpoint_broke(
            urllib.error.URLError(TimeoutError("timed out"))))

    def test_dns_error_false(self) -> None:
        self.assertFalse(_proxy_endpoint_broke(OSError(101, "getaddrinfo failed")))
        self.assertFalse(_proxy_endpoint_broke(
            urllib.error.URLError(OSError(101, "getaddrinfo failed"))))

    def test_value_error_false(self) -> None:
        self.assertFalse(_proxy_endpoint_broke(ValueError("bad url")))


class TestLoopbackClassifier(unittest.TestCase):
    def test_loopback_true(self) -> None:
        self.assertTrue(ups._is_loopback("http://127.0.0.1:7892"))
        self.assertTrue(ups._is_loopback("127.0.0.1:7892"))
        self.assertTrue(ups._is_loopback("http://localhost:7897"))
        self.assertTrue(ups._is_loopback("socks://[::1]:1080"))

    def test_remote_false(self) -> None:
        self.assertFalse(ups._is_loopback("http://192.0.2.1:8080"))
        self.assertFalse(ups._is_loopback("proxy.corp.example:8080"))


if __name__ == "__main__":
    unittest.main()
