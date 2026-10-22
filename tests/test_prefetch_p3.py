"""P3 网页闭环测试：SSRF 防护、econ_gate、WebFetcher 失败注入、双栏阅读器。

SSRF 用例覆盖方案 §10.2-4 的全部拒绝清单；
WebFetcher 的 HTTP 行为用本地 http.server 在 loopback 上驱动
（fetch 传 validate=False 仅为测试注入，生产路径恒为 True，
SSRF 拒绝清单由 TestSSRF 直接覆盖）。
"""

from __future__ import annotations

import http.server
import socket
import threading
import unittest
from unittest import mock

from pingyitong.services.prefetch.econ_gate import EconGate, SiteStats, threshold
from pingyitong.services.prefetch.ssrf import validate_url
from pingyitong.services.prefetch.web_fetcher import (
    fetch,
)


class TestSSRF(unittest.TestCase):
    """§10.2-4 拒绝清单：全部必须拒绝。"""

    def test_loopback_rejected(self) -> None:
        self.assertIn("loopback", validate_url("http://127.0.0.1/x") or "")

    def test_private_ranges_rejected(self) -> None:
        for url in ("http://10.0.0.1/", "http://172.16.0.1/",
                    "http://192.168.1.1/"):
            self.assertIsNotNone(validate_url(url), url)

    def test_cgnat_rejected(self) -> None:
        """CGNAT 100.64.0.0/10（RFC 6598）必须单独拦。

        Python 3.11 的 ``is_private``/``is_reserved`` 都不覆盖该段
        （实测均返回 False），不显式判断就会被当作公网地址放行。
        """
        for ip in ("100.64.0.1", "100.100.50.7", "100.127.255.254"):
            with self.subTest(ip=ip):
                self.assertIn("CGNAT", validate_url(f"http://{ip}/x") or "")
        # 段外地址不得误伤（前后边界各取一个）
        for ip in ("100.63.255.255", "100.128.0.1"):
            with self.subTest(ip=ip):
                self.assertIsNone(validate_url(f"http://{ip}/x"))

    def test_file_scheme_rejected(self) -> None:
        self.assertIn("协议", validate_url("file:///etc/passwd"))

    def test_ftp_scheme_rejected(self) -> None:
        self.assertIsNotNone(validate_url("ftp://example.com/file"))

    def test_data_scheme_rejected(self) -> None:
        self.assertIsNotNone(validate_url("data:text/html,hi"))

    def test_no_credentials(self) -> None:
        self.assertIn("凭据", validate_url("http://user:pass@example.com/"))

    def test_ipv6_ula_rejected(self) -> None:
        self.assertIsNotNone(validate_url("http://[fc00::1]/"))

    def test_dns_failure_rejected(self) -> None:
        """DNS 解析失败必须拒绝。

        mock 网络层而非依赖真实 DNS：本机代理软件的 fake-ip 模式会把
        任意域名（含不存在的）解析到 198.18.0.0/15，"不存在的域名解析
        失败"这一前提在真实 DNS 上不成立。
        """
        with mock.patch("socket.getaddrinfo", side_effect=socket.gaierror(-2, "name unknown")):
            self.assertIn("解析失败", validate_url("http://this-host-does-not-exist-xyz.invalid/"))

    def test_public_host_allowed(self) -> None:
        """公网域名解析到公网地址应放行（mock 网络层，测放行分支）。"""
        fake = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP,
                 "", ("93.184.216.34", 443))]
        with mock.patch("socket.getaddrinfo", return_value=fake):
            self.assertIsNone(validate_url("https://www.python.org/"))

    def test_fake_ip_hijack_rejected(self) -> None:
        """域名被劫持解析到 198.18.0.0/15（RFC 2544 基准段，代理软件
        fake-ip 的默认网段）必须拒绝——可解析不等于安全。"""
        fake = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP,
                 "", ("198.18.0.15", 80))]
        with mock.patch("socket.getaddrinfo", return_value=fake):
            reason = validate_url("http://anything.example/")
        self.assertIsNotNone(reason, "fake-ip 段必须被拒绝")
        self.assertIn("198.18.0.15", reason or "")


class TestEconGate(unittest.TestCase):
    def test_threshold_formula(self) -> None:
        """a* = C / (ΔL·V + C)：C=1, ΔL=2, V=1 → 1/3。"""
        self.assertAlmostEqual(threshold(1.0, 2.0, 1.0), 1 / 3, places=3)

    def test_good_site_allowed(self) -> None:
        gate = EconGate(lambda site: SiteStats(prefetched=50, used=45))
        self.assertTrue(gate.allow_prefetch("good.com"))

    def test_bad_site_blocked(self) -> None:
        gate = EconGate(lambda site: SiteStats(prefetched=50, used=5))
        self.assertFalse(gate.allow_prefetch("bad.com"))

    def test_coldstart_quota(self) -> None:
        """冷启动：无历史时给试探配额，耗尽即停。"""
        gate = EconGate(lambda site: SiteStats(), coldstart_per_site_day=3)
        for _ in range(3):
            self.assertTrue(gate.allow_prefetch("new.com"))
        self.assertFalse(gate.allow_prefetch("new.com"))
        # 其他站点不受影响
        self.assertTrue(gate.allow_prefetch("other.com"))


class _Handler(http.server.BaseHTTPRequestHandler):
    """测试站点：/ok /login /loop /self_redirect /noloc /robots_disallow /big /next"""

    robots_disallow = False

    def log_message(self, *args):  # 静默
        pass

    def do_GET(self):
        if self.path == "/robots.txt":
            if self.robots_disallow:
                body = b"User-agent: *\nDisallow: /\n"
            else:
                body = b"User-agent: *\nAllow: /\n"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/ok":
            html = ("<html><head><title>测试页</title></head><body>"
                    "<p>第一段正文内容，足够长以被抽取。</p>"
                    "<p>第二段正文内容。</p>"
                    "<a href='/page2'>下一页</a></body></html>")
            self._html(html)
        elif self.path == "/login":
            html = ("<html><body><form>"
                    "<input type='password' name='pw'></form></body></html>")
            self._html(html)
        elif self.path == "/loop":
            self.send_response(302)
            self.send_header("Location", "http://10.0.0.5/inner")
            self.end_headers()
        elif self.path == "/self_redirect":
            self.send_response(302)
            self.send_header("Location", "/ok")
            self.end_headers()
        elif self.path == "/noloc":
            self.send_response(302)
            self.end_headers()
        elif self.path == "/401":
            self.send_response(401)
            self.end_headers()
        elif self.path == "/big":
            self._html("<p>" + "x" * (3 * 1024 * 1024) + "</p>")
        else:
            self.send_response(404)
            self.end_headers()

    def _html(self, content: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))


class TestWebFetcherHTTP(unittest.TestCase):
    """HTTP 行为：登录墙 / 截断 / 重定向 / robots（loopback + validate=False 注入）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        from pingyitong.services.prefetch.web_fetcher import clear_robots_cache

        clear_robots_cache()

    def test_ok_and_next_candidates(self) -> None:
        result = fetch(f"{self.base}/ok", validate=False)
        self.assertEqual(result.kind, "ok")
        article = __import__(
            "pingyitong.services.prefetch.web_fetcher", fromlist=["extract_article"]
        ).extract_article(result.html, result.final_url, result.etag)
        self.assertTrue(article.text_blocks)
        self.assertEqual(len(article.next_candidates), 1)

    def test_login_wall_blocked(self) -> None:
        result = fetch(f"{self.base}/login", validate=False)
        self.assertEqual(result.kind, "blocked")
        self.assertIn("登录", result.reason)

    def test_401_blocked(self) -> None:
        result = fetch(f"{self.base}/401", validate=False)
        self.assertEqual(result.kind, "blocked")

    def test_truncation(self) -> None:
        """超过单页上限流式截断，不抛异常。"""
        result = fetch(f"{self.base}/big", validate=False, max_bytes=64 * 1024)
        self.assertEqual(result.kind, "ok")
        self.assertLessEqual(len(result.html.encode("utf-8")), 64 * 1024 + 4)

    def test_robots_disallow_blocked(self) -> None:
        from pingyitong.services.prefetch.web_fetcher import clear_robots_cache

        _Handler.robots_disallow = True
        try:
            clear_robots_cache()
            result = fetch(f"{self.base}/ok", validate=False)
            self.assertEqual(result.kind, "blocked")
            self.assertIn("robots", result.reason)
        finally:
            _Handler.robots_disallow = False
            clear_robots_cache()

    def test_redirect_to_private_blocked(self) -> None:
        """/loop 302 跳 10.0.0.5（内网）——重定向逐跳校验必须拦截。"""
        result = fetch(f"{self.base}/loop", validate=False)
        self.assertEqual(result.kind, "blocked")
        self.assertIn("重定向", result.reason)

    def test_redirect_to_loopback_blocked_even_without_validate(self) -> None:
        """validate=False 只放宽初始 URL：302 跳回 loopback 仍逐跳拦截。"""
        result = fetch(f"{self.base}/self_redirect", validate=False)
        self.assertEqual(result.kind, "blocked")
        self.assertIn("重定向", result.reason)

    def test_redirect_missing_location_error(self) -> None:
        """/noloc 302 不带 Location——返回错误原因，不静默跟进。"""
        result = fetch(f"{self.base}/noloc", validate=False)
        self.assertEqual(result.kind, "error")
        self.assertIn("Location", result.reason)


class TestReaderView(unittest.TestCase):
    """双栏阅读器：set_blocks 原文先行，update_translation 逐块替换。"""

    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_blocks_and_translation(self) -> None:
        from PyQt6.QtGui import QPixmap

        from pingyitong.services.prefetch.reader_view import ReaderView
        from pingyitong.ui.design.theme import build_theme

        view = ReaderView(build_theme("dark"))
        view.resize(800, 400)
        view.set_blocks(["第一段", "第二段"])
        self.assertEqual(view.block_count(), 2)
        view.update_translation(0, "First paragraph")
        self.assertEqual(view._rows[0].translated_label.text(), "First paragraph")
        # 离屏渲染（纯文本，无脚本执行）
        pix = QPixmap(view.size())
        view.render(pix)
        view.deleteLater()

    def test_markup_is_displayed_as_literal_source_and_translation(self) -> None:
        from PyQt6.QtCore import Qt

        from pingyitong.services.prefetch.reader_view import ReaderView
        from pingyitong.ui.design.theme import build_theme

        view = ReaderView(build_theme("dark"))
        view.set_blocks(["<b>Save</b> {{name}}"])
        view.update_translation(0, "<b>保存</b> {{name}}")
        self.assertEqual(view._rows[0].source_label.textFormat(), Qt.TextFormat.PlainText)
        self.assertEqual(view._rows[0].translated_label.textFormat(), Qt.TextFormat.PlainText)
        view.deleteLater()


class TestPrefetchStopSwitch(unittest.TestCase):
    """预取中途停止开关：用户关闭联网预翻译必须中止当前页面的逐块翻译。"""

    @staticmethod
    def _app_with_prefetch(enabled: bool):
        from types import SimpleNamespace

        from pingyitong.ui.app import Application

        app = Application.__new__(Application)
        app.settings = SimpleNamespace(config=SimpleNamespace(
            prefetch=SimpleNamespace(enabled=enabled),
        ))
        return app

    def test_disabled_prefetch_requests_stop(self) -> None:
        app = self._app_with_prefetch(enabled=False)
        self.assertTrue(app._stop_prefetch_requested)

    def test_enabled_prefetch_keeps_running(self) -> None:
        app = self._app_with_prefetch(enabled=True)
        self.assertFalse(app._stop_prefetch_requested)

    def test_unreadable_config_conservatively_continues(self) -> None:
        """配置暂不可读时不得误停（保守继续，等待配置恢复）。"""
        from types import SimpleNamespace

        from pingyitong.ui.app import Application

        app = Application.__new__(Application)
        app.settings = SimpleNamespace(config=None)
        self.assertFalse(app._stop_prefetch_requested)


if __name__ == "__main__":
    unittest.main()
