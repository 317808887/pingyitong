"""网页对照阅读器对话框（联网预翻译 P3）。

对应 D2 的体验边界：粘贴 URL → 内置纯文本双栏阅读器；
浏览器里的正常浏览仍走屏幕监视翻译。

流程（网络/翻译在后台线程，UI 更新经信号队列回主线程）：
  1. fetch（robots/SSRF/登录墙/截断，见 web_fetcher）；
  2. trafilatura 抽取正文块 → 双栏视图先显示原文；
  3. 逐块走正式翻译服务（缓存、术语与输出校验）→ 写预取缓存 → 视图替换；
  4. 下一章候选入队调度器（NEXT_CHAPTER，受盈亏门控）。

门控语义：总开关关 / 离线 / 配额尽 → 不抓取，视图显示降级提示；
抓取失败 → 显示原因并提示"已回退屏幕监视翻译"。
"""

from __future__ import annotations

import threading
from copy import deepcopy

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ..services.language import detect_language
from ..services.prefetch.reader_view import ReaderView
from ..services.prefetch.ssrf import validate_url
from ..services.prefetch.web_fetcher import extract_article, fetch
from ..utils.logger import get_logger

logger = get_logger(__name__)


class WebReaderDialog(QDialog):
    """网页对照阅读器（非模态，可常开）。"""

    #: 后台线程 → 主线程的 UI 任务投递（payload 为无参 callable）
    ui_task = pyqtSignal(object)

    def __init__(self, application, parent=None) -> None:
        super().__init__(parent or None)
        self._app = application
        self._config = application.settings.config
        self._busy = False
        self._blocks: list[str] = []
        self._host = ""
        self._translation_cancel = threading.Event()

        self.setWindowTitle("屏译通 · 网页翻译")
        self.setModal(False)
        self.resize(980, 720)
        self._build_ui()
        self.ui_task.connect(self._run_ui_task)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        url_row = QHBoxLayout()
        self._url_input = QLineEdit()
        self._url_input.setPlaceholderText(
            "输入网页地址（仅静态页面；登录墙/JS 渲染页将回退屏幕翻译）"
        )
        self._url_input.returnPressed.connect(self._on_go)
        self._go_button = QPushButton("翻译此页")
        self._go_button.clicked.connect(self._on_go)
        url_row.addWidget(self._url_input, 1)
        url_row.addWidget(self._go_button)
        layout.addLayout(url_row)

        self._badge = QLabel(self._app.prefetch_gate.snapshot().badge)
        self._badge.setObjectName("Secondary")
        self._status = QLabel("")
        self._status.setObjectName("Secondary")
        self._status.setWordWrap(True)
        layout.addWidget(self._badge)
        layout.addWidget(self._status)

        self._reader = ReaderView(self._app.theme)
        layout.addWidget(self._reader, 1)

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def open_url(self, url: str) -> None:
        """外部入口：托盘/热键带 URL 直接打开。"""
        self.show()
        self.raise_()
        self.activateWindow()
        if url:
            self._url_input.setText(url)
            self._on_go()

    def _on_go(self) -> None:
        if self._busy:
            return
        url = self._url_input.text().strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        gate = self._app.prefetch_gate
        snapshot = gate.snapshot()
        self._badge.setText(snapshot.badge)

        if not snapshot.allowed:
            # D1：缓存直读仍会发生（见 _translate_blocks），但抓取必须停止
            self._status.setText(
                f"当前模式「{snapshot.badge}」：不抓取网页。"
                "已回退屏幕监视翻译；已缓存的网页译文打开时仍可读。"
            )
            return

        ssrf_reason = validate_url(url)
        if ssrf_reason:
            self._status.setText(f"该地址不允许抓取：{ssrf_reason}")
            return

        self._busy = True
        self._translation_cancel.set()
        self._translation_cancel = threading.Event()
        cancellation = self._translation_cancel
        self._config = deepcopy(self._app.settings.config)
        self._go_button.setEnabled(False)
        self._status.setText("抓取页面…")
        host = url.split("/")[2] if "//" in url else url
        self._host = host

        def worker() -> None:
            result = fetch(
                url,
                timeout_s=max(5.0, self._config.prefetch.probe_timeout_ms / 1000 * 5),
                max_bytes=self._config.prefetch.max_page_bytes,
            )
            if result.kind != "ok":
                self._app.prefetch_gate.notify_fetch_failed(
                    host, result.reason or result.kind
                )
                self._post_to_ui(
                    lambda r=result: self._status.setText(
                        f"抓取失败（{r.reason}）——已回退屏幕监视翻译"
                    )
                )
                self._done()
                return
            self._app.prefetch_gate.notify_fetch_ok(host)
            article = extract_article(result.html, result.final_url, result.etag)
            if not article.text_blocks:
                self._post_to_ui(
                    lambda: self._status.setText(
                        "页面正文抽取为空（疑似 JS 渲染页），已回退屏幕监视翻译"
                    )
                )
                self._done()
                return
            self._post_to_ui(
                lambda: self._populate(article) if not cancellation.is_set() else None
            )

        threading.Thread(target=worker, name="web-fetch", daemon=True).start()

    # ------------------------------------------------------------------
    # 抓取成功后的填充与翻译
    # ------------------------------------------------------------------

    def _populate(self, article) -> None:
        self._blocks = list(article.text_blocks)
        self._reader.set_blocks(self._blocks)
        self._status.setText(f"《{article.title}》{len(self._blocks)} 块，正在翻译…")
        self._done()
        self._translate_blocks(article)

    def _translate_blocks(self, article) -> None:
        """Use the verified service and discard results from superseded settings."""
        from ..services.prefetch.sentence_cache import SentenceEntry

        previous = getattr(self, "_translation_cancel", None)
        if previous is not None:
            previous.set()
        cancellation = threading.Event()
        self._translation_cancel = cancellation
        self._config = config = deepcopy(self._app.settings.config)
        translator = self._app.translator
        revision = translator._request_revision
        namespace = translator._live_namespace()
        target = config.translation.target_language
        source = config.translation.source_language
        blocks = tuple(self._blocks)
        host = self._host
        self._busy = True
        self._go_button.setEnabled(False)

        def current() -> bool:
            return (
                not cancellation.is_set()
                and self._translation_cancel is cancellation
                and self._app.settings.config == config
                and translator._request_revision == revision
            )

        def publish(index: int, translated: str) -> None:
            if current():
                self._reader.update_translation(index, translated)

        def worker() -> None:
            completed = 0
            try:
                for index, block in enumerate(blocks):
                    if not current():
                        return
                    try:
                        translated = translator.translate_text(block)
                    except Exception:  # noqa: BLE001 - one block cannot stall the reader
                        logger.warning("网页段落翻译失败，保留原文", exc_info=True)
                        continue
                    if not current():
                        return
                    # translate_text deliberately returns the real source on
                    # an unrecoverable refusal/language failure. It remains
                    # visible in the left column, never as a cached translation.
                    if not translated or not translator._valid_output(block, translated):
                        continue
                    completed += 1
                    self._post_to_ui(
                        lambda i=index, t=translated: publish(i, t)
                    )
                    # The main translation cache scopes online providers and
                    # concise output. The prefetch schema has no such fields;
                    # only standard local results belong in that shared cache.
                    if (
                        not namespace.startswith("api:")
                        and namespace == translator._live_namespace()
                        and not config.translation.concise_output
                        and current()
                    ):
                        detected = source if source != "auto" else detect_language(
                            block, config.translation.fallback_source_language
                        ).code
                        try:
                            self._app.prefetch_cache.put_many([SentenceEntry(
                                source=block, translated=translated, src_lang=detected,
                                tgt_lang=target, domain=host,
                                model_version=self._app.prefetch_cache._model_version,
                                source_kind="web", ttl_days=config.prefetch.site_ttl_days,
                                etag=article.etag,
                            )])
                        except Exception:  # noqa: BLE001 - cache failure does not lose the answer
                            logger.warning("网页译文缓存写入失败", exc_info=True)
            finally:
                def finish() -> None:
                    if self._translation_cancel is not cancellation:
                        return
                    self._busy = False
                    self._go_button.setEnabled(True)
                    if current():
                        self._status.setText(
                            "翻译完成" if completed == len(blocks) else
                            f"已翻译 {completed}/{len(blocks)} 块，未完成的段落保留原文"
                        )
                        self._enqueue_next_chapters(article)
                    elif not cancellation.is_set():
                        self._status.setText("翻译设置已变化，请重新翻译此页")

                self._post_to_ui(finish)

        threading.Thread(target=worker, name="web-translate", daemon=True).start()

    def _enqueue_next_chapters(self, article) -> None:
        """下一章候选入队（NEXT_CHAPTER 优先级，受盈亏门控 + 冷启动配额）。"""
        from ..services.prefetch.scheduler import PrefetchJob, Priority

        gate = self._app.prefetch_econ_gate
        enqueued = 0
        for candidate in article.next_candidates[
            : self._config.prefetch.prefetch_next_max
        ]:
            if not gate.allow_prefetch(self._host):
                break
            self._app.prefetch_scheduler.enqueue(PrefetchJob(
                priority=Priority.NEXT_CHAPTER, seq=0, kind="web_page",
                track_or_site=self._host, payload=candidate,
                page_key=candidate,
            ))
            enqueued += 1
        if enqueued:
            logger.info("下一章预取入队：%s 个（%s）", enqueued, self._host)

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------

    def _post_to_ui(self, fn) -> None:
        """后台线程 → 主线程投递（信号队列连接保证在 UI 线程执行）。"""
        self.ui_task.emit(fn)

    def _run_ui_task(self, fn) -> None:
        fn()

    def _done(self) -> None:
        def _finish() -> None:
            self._busy = False
            self._go_button.setEnabled(True)
        self._post_to_ui(_finish)

    def closeEvent(self, event) -> None:
        # Workers only inspect this thread-safe event; no Qt queries off-thread.
        self._translation_cancel.set()
        super().closeEvent(event)
