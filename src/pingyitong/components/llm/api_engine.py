"""在线 AI 翻译引擎（OpenAI 兼容协议，自研）。

任何提供 OpenAI ``/chat/completions`` 兼容接口的服务（DeepSeek、智谱 GLM、
月之暗面 Kimi、阿里通义、OpenAI、本机 Ollama 等）都能直接接入：
填接口地址、API Key、模型名三项即可，不需要为每家服务商写适配器。

设计要点：

1. **复用既有批量协议**——引擎只负责"把一段提示词发出去、把回包原文
   取回来"，片段分隔符（``<<<SEG>>>``）、术语占位符、对照模式等约定
   全部由上游的 prompt 工程与解析层处理。在线大模型对分隔符指令的
   遵循度远高于 1.7B 本地模型，批量对位基本不失配。

2. **连接复用**——桌面翻译的请求特点是小而频繁（单批几十到几百字符），
   HTTPS 握手（TCP + TLS，实测 100–300ms）会占单次请求耗时的一半以上。
   因此引擎维护一个小型空闲连接池：请求结束把连接归还，下次直接复用；
   服务端提前断开导致的"陈旧连接"错误自动换新连接重试一次。

3. **故障冷却**——云端不可达（断网、Key 失效、限流）时，若每个片段
   都要等一次超时再回退本地，整帧延迟会被拖到不可用。连续失败达到
   阈值后引擎进入冷却期（默认 60 秒）：期间 :meth:`is_available` 返回
   ``False``，翻译请求**直接**走本地模型，不再白付超时；冷却期满自动
   恢复尝试。其中认证/权限/路径类错误（401/403/404）重试必然复现，
   **首次失败即进入冷却**——同一帧的后续片段不必再逐段白付往返。

4. **思考过程剥离**——推理类模型（DeepSeek-R1、Qwen-thinking 等）可能在
   ``content`` 里内联 ``<think>`` 段；译文里出现思考过程是用户可见的
   质量事故，返回前一律剥掉。

5. **输出截断防护**——响应 ``finish_reason="length"`` 表示输出在预算处
   被掐断，拿到的是半截译文。半截句子画到屏幕上是可见的质量事故，
   因此按翻倍预算自动重问一次；仍截断则按失败处理，由调用方回退
   本地模型产出完整译文（宁可慢一帧，不要半句话）。

6. **内容分片兼容**——部分 OpenAI 兼容网关把 ``message.content`` 返回为
   分片数组（``[{"type": "text", ...}]``）而非字符串；不解这个形态会把
   成功响应误判成空内容，连续几次后触发冷却、整个在线模式白退本地。

安全说明：请求发往用户显式配置的接口地址，携带用户自己填写的 Key；
除该地址外引擎不访问任何其他主机。Key 明文存于本机配置文件。
"""

from __future__ import annotations

import contextlib
import http.client
import json
import re
import select
import ssl
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from ...utils.logger import get_logger
from .base import EngineStatus

logger = get_logger(__name__)

#: 连续失败多少次后进入冷却期
FAILURE_THRESHOLD = 3
#: 冷却时长（秒）：期间请求直接走本地模型
COOLDOWN_SECONDS = 60.0
#: 首次失败即触发冷却的确定性错误：重试必然复现，不值得再付往返。
#: 400（请求被拒：模型名/参数写错的最常见表现）与 401/403/404 同理；
#: 402（余额不足/配额耗尽，如 DeepSeek 欠费）在分钟尺度内同样不会自愈。
#: 429（限流）与 5xx 可能是瞬时的，仍按连续失败计数。
DETERMINISTIC_FAILURE_STATUSES = frozenset({400, 401, 402, 403, 404})

#: 单次请求输出 token 的硬顶，与 settings 对 ``api.max_tokens`` 的钳制上限
#: 一致（translation_service 的在线输出上限同值）。截断重试把预算翻倍后
#: 不得越过它——再高服务商大概率直接 400，反而把一次成功的请求拖进冷却。
MAX_OUTPUT_TOKENS = 8192

# OpenAI 推理系（o 系列/gpt-5 等）的参数怪癖：不接受 ``max_tokens`` 参数名
# （400 并提示改用 ``max_completion_tokens``），部分模型只接受默认温度。
# 这类 400 重试必然复现，但错误信息里给出了正确参数名——按提示学习一次
# 自适应重试，不让"填了新模型名"被误报成"配置或认证错误"而冷却。
_COMPLETION_TOKENS_HINT = "max_completion_tokens"
_TEMPERATURE_HINT = "'temperature'"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
#: 未闭合的 <think>：模型在 max_tokens 内没写完思考，正文不可信，整段剥离
_THINK_UNCLOSED = re.compile(r"<think>.*$", re.DOTALL)


# ------------------------------------------------------------------
# 系统代理（与 urlopen_proxy_safe.py 语义保持一致）
# ------------------------------------------------------------------

def _proxy_entries() -> dict[str, str]:
    """当前系统代理映射（http/https 条目）；无代理返回空 dict。"""
    try:
        proxies = urllib.request.getproxies()
    except Exception:  # noqa: BLE001 - 注册表/环境异常不拦住连接
        return {}
    return {k: v for k, v in proxies.items() if k in ("http", "https")}


def _is_loopback(proxy_value: str) -> bool:
    """代理论送地址形如 ``http://127.0.0.1:7892``；判断其主机段。"""
    host = urlsplit(proxy_value if "//" in proxy_value else "//" + proxy_value).hostname
    if not host:
        return False
    return host.lower() in {"127.0.0.1", "localhost", "::1", "[::1]"}


def _system_proxy_for(url: str) -> str | None:
    """返回适用于给定 URL 的系统代理地址；无代理返回 ``None``。

    远程代理与本机代理都返回，调用方自行决定是否走代理或降级直连。
    """
    proxies = _proxy_entries()
    if not proxies:
        return None
    scheme = urlsplit(url).scheme or "https"
    return proxies.get(scheme)


def _new_connection_through_proxy(
    proxy: str,
    target_host: str,
    target_port: int,
    scheme: str,
    timeout: float,
    ssl_context: ssl.SSLContext | None,
) -> http.client.HTTPConnection:
    """通过 HTTP 代理建到目标主机的连接。

    HTTPS 目标：先 CONnect 隧道，再在 socket 上做 SSL 握手；
    HTTP 目标：直接复用代理连接（http.client 的请求行会带完整 URL）。
    """
    proxy_parts = urlsplit(proxy if "//" in proxy else "//" + proxy)
    proxy_host = proxy_parts.hostname or ""
    proxy_port = proxy_parts.port or (443 if proxy_parts.scheme == "https" else 80)

    conn = http.client.HTTPConnection(proxy_host, proxy_port, timeout=timeout)
    conn.connect()

    if scheme == "https":
        # HTTP CONNECT 隧道：代理把 TCP 连接转发到目标主机
        connect_request = (
            f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
            f"Host: {target_host}:{target_port}\r\n"
            "\r\n"
        )
        conn.send(connect_request.encode("utf-8"))
        resp = conn.response()
        if resp.status != 200:
            conn.close()
            raise OSError(f"代理 CONNECT 失败：HTTP {resp.status}")
        resp.read()
        sock = ssl_context.wrap_socket(conn.sock, server_hostname=target_host)
        conn.sock = sock

    return conn


def strip_thinking(text: str) -> str:
    """剥离推理模型内联在正文中的 ``<think>`` 段。"""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_UNCLOSED.sub("", text)
    return text.strip()


class ApiEngine:
    """OpenAI 兼容的在线翻译引擎，实现 :class:`TranslationEngine` 协议。

    无本地权重、无加载过程：``load`` 恒成功、``warm_up`` 为空操作
    （启动时发探测请求只会白花用户配额）。
    """

    def __init__(self, api_config: object, model_id: str = "api") -> None:
        self.model_id = model_id
        self._base_url = str(getattr(api_config, "base_url", "")).strip().rstrip("/")
        self._api_key = str(getattr(api_config, "api_key", "")).strip()
        self._model_name = str(getattr(api_config, "model_name", "")).strip()
        self._temperature = float(getattr(api_config, "temperature", 0.2))
        self._max_tokens = int(getattr(api_config, "max_tokens", 768))
        self._timeout = max(1.0, float(getattr(api_config, "timeout_seconds", 10.0)))
        # 预建连接数与批并发对齐：整帧多批并发时每条新连接都要现场付一次
        # 100–300ms 握手，首帧里只预热一条等于把其余批次的握手留在关键路径上。
        self._preconnect_target = max(
            1, min(int(getattr(api_config, "max_concurrency", 4) or 4), 8)
        )

        parts = urlsplit(self._base_url)
        self._scheme = parts.scheme or "https"
        self._host = parts.hostname or ""
        port = parts.port
        if port is None:
            port = 443 if self._scheme == "https" else 80
        self._port = port
        # 接口路径前缀（如 /v1）；根地址无路径时为空串
        self._prefix = (parts.path or "").rstrip("/")

        self._pool_lock = threading.Lock()
        self._idle_connections: list[http.client.HTTPConnection] = []
        self._closed = False
        self._metrics_lock = threading.Lock()
        self._last_latency = 0.0
        self._total_calls = 0
        # Cache entries produced by a local fallback must not hide the next
        # online attempt. Failures and explicit connection recovery advance
        # this generation; ordinary successful translations remain reusable.
        self._failure_generation = 0
        # A successful connection test starts a new recovery generation. Old
        # in-flight responses must not undo recovery or newer failure state.
        self._recovery_generation = 0
        self._consecutive_failures = 0
        self._last_failure_at = 0.0
        self._last_error_status: int | None = None
        # 400 自适应学习标志（并发下为幂等布尔赋值，无需加锁）
        self._use_completion_tokens = False
        self._omit_temperature = False

        if self._scheme not in {"http", "https"} or not self._host:
            logger.warning("在线 AI 接口地址无效：%r", self._base_url)
            self._scheme = ""
            self._host = ""

        # TLS 上下文全引擎复用：创建一次要加载系统 CA 证书（实测 ~10ms），
        # 放在每次新建连接里等于每条新连接都白付；SSLContext 官方保证
        # 跨线程共享安全，并发建连各自 create SSL socket 互不干扰。
        self._ssl_context = (
            ssl.create_default_context() if self._scheme == "https" else None
        )

    # ------------------------------------------------------------------
    # 协议实现
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """三项必填信息（地址/模型名）是否完整。Key 可留空。"""
        return bool(self._host and self._scheme and self._model_name)

    def is_available(self) -> bool:
        return self.is_configured() and not self._in_cooldown()

    def is_loaded(self) -> bool:
        """在线引擎无加载概念，配置完整即视为就绪。"""
        return self.is_configured()

    def load(self) -> bool:
        return self.is_configured()

    def unload(self) -> None:
        """关闭空闲连接；在途请求归还连接时也一并关闭。"""
        with self._pool_lock:
            self._closed = True
            idle, self._idle_connections = self._idle_connections, []
        for conn in idle:
            with contextlib.suppress(Exception):
                conn.close()

    def warm_up(self) -> None:
        """空操作：在线引擎没有首次前向计算的开销，无需预热。

        也不发探测请求——那会消耗用户的 API 配额。
        """

    def preconnect(self) -> None:
        """后台并行预建若干条 TCP(+TLS) 连接并入池（不发送任何 HTTP 请求）。

        桌面翻译的第一帧往往在启用在线 AI 后立刻到来，首帧若还要现付
        100–300ms 的握手，"在线更快"就打了个折扣；把握手提前到配置生效
        时刻并行完成，首个请求直接复用现成连接。数量与 ``max_concurrency``
        对齐——整帧多批并发时每条新连接都是一次独立握手，只预热一条的话
        其余批次的握手仍落在首帧关键路径上。纯连接不产生请求，
        **不消耗用户配额**；连接在首个请求前被服务端关掉的情况由
        :meth:`_checkout` 的陈旧检出兜底（检出成本≈0，换新连接即可）。
        """
        if not self.is_configured() or self._closed:
            return
        if self._preconnect_target <= 1:
            self._preconnect_one()
            return
        with ThreadPoolExecutor(
            max_workers=self._preconnect_target, thread_name_prefix="ApiPreconnect"
        ) as executor:
            futures = [executor.submit(self._preconnect_one) for _ in range(self._preconnect_target)]
            for future in futures:
                # 预连接是尽力而为：单条失败已在 _preconnect_one 内静默，
                # 这里只需等全部完成（with 退出时隐式 join），不抛给调用方
                with contextlib.suppress(Exception):
                    future.result()

    def _preconnect_one(self) -> None:
        """预建一条连接入池；失败静默（首次真实请求会再试）。"""
        if not self.is_configured() or self._closed:
            return
        try:
            conn = self._new_connection()
            conn.connect()
        except OSError as exc:
            # 预连接失败不打扰用户：首次真实请求会原样再试并把错误
            # 走正常失败路径（冷却/回退本地），这里只留调试日志
            logger.debug("在线 AI 预连接未就绪（首次请求时再试）：%s", exc)
            return
        self._checkin(conn)

    def translate_batch(
        self,
        segments: list[str],
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> list[str]:
        """翻译一批片段。

        上游（:meth:`ModelPool.translate`）始终把已拼装好的 user 提示词
        作为单元素列表传入；列表长度大于 1 时按分隔符规则拼装兜底。
        返回与输入等长的列表；请求失败时元素为空字符串，
        由调用方回退本地模型。
        """
        if len(segments) == 1:
            user_text = segments[0]
        else:
            from ...constants import SEGMENT_DELIMITER

            user_text = f"\n{SEGMENT_DELIMITER}\n".join(segments)

        started = time.perf_counter()
        generation = self._request_generation()
        try:
            content = self._chat(
                system=system_prompt,
                user=user_text,
                max_tokens=max_tokens or self._max_tokens,
            )
        except ApiRequestError as exc:
            self._record_failure(status=exc.status, generation=generation)
            logger.debug("在线 AI 请求失败：HTTP %s", exc.status)
            return [""]
        except Exception as exc:  # noqa: BLE001 - 网络层异常统一按失败处理
            self._record_failure(generation=generation)
            logger.debug("在线 AI 请求失败：%s", exc)
            return [""]

        content = strip_thinking(content or "").strip()
        if not content:
            # 空响应按失败记账而非成功：推理类模型（DeepSeek-R1、
            # Qwen-thinking）把 max_tokens 耗在思考上时 content 为空，
            # 这类场景是系统性的——记成功会让每个片段永远白付一次
            # 云端往返再回退本地，冷却机制永远不会触发。
            self._record_failure(generation=generation)
            logger.warning(
                "在线 AI 返回空内容（模型可能把输出额度耗在思考上），按失败记账"
            )
            return [""]
        self._record_success(time.perf_counter() - started, generation=generation)
        return [content]

    def status(self) -> EngineStatus:
        available = self.is_available()
        if not self.is_configured():
            message = "未配置接口地址或模型名"
        elif not available:
            if self._last_error_status in DETERMINISTIC_FAILURE_STATUSES:
                message = (
                    f"配置或认证错误（HTTP {self._last_error_status}），"
                    f"{COOLDOWN_SECONDS:.0f}s 内走本地模型"
                )
            else:
                message = f"连续失败 {FAILURE_THRESHOLD} 次，{COOLDOWN_SECONDS:.0f}s 后重试"
        else:
            message = self._model_name
        return EngineStatus(
            model_id=self.model_id,
            loaded=True,
            available=available,
            message=message,
            path=self._base_url,
            last_latency=self._last_latency,
            total_calls=self._total_calls,
        )

    @property
    def failure_generation(self) -> int:
        """Monotonic cache generation for failures and explicit recovery."""
        with self._metrics_lock:
            return self._failure_generation

    # ------------------------------------------------------------------
    # UI 测试连接
    # ------------------------------------------------------------------

    def test(self) -> tuple[bool, str]:
        """连通性自检：发送一条最小请求，返回 ``(是否成功, 描述)``。

        供设置面板的"测试连接"按钮使用；成功描述含往返耗时，
        失败描述含服务商返回的错误信息（如 401 无效 Key）。
        """
        if not self.is_configured():
            return False, "请先填写接口地址与模型名称"
        started = time.perf_counter()
        generation = self._request_generation()
        try:
            content = self._chat(
                system="You are a translator. Output only the translation.",
                user="Translate to Chinese: OK",
                # 与生产路径同预算：自检结果要能反映真实请求的可行性。
                # 曾硬编码 16——推理类模型的思考就能耗尽它，自检报"返回
                # 为空"，而实际翻译请求（预算 768）其实正常。
                max_tokens=self._max_tokens,
            )
        except ApiRequestError as exc:
            self._record_failure(status=exc.status, generation=generation)
            return False, _friendly_error(exc)
        except Exception as exc:  # noqa: BLE001 - 错误信息直接展示给用户
            self._record_failure(generation=generation)
            return False, _friendly_error(exc)
        elapsed_seconds = time.perf_counter() - started
        elapsed = elapsed_seconds * 1000
        reply = strip_thinking(content or "").strip()
        if not reply:
            self._record_failure(generation=generation)
            return False, (
                "连接正常，但模型返回为空——请确认模型名是否为翻译模型；"
                "推理类模型（如 DeepSeek-R1）的思考会耗尽输出额度，"
                "可调大\"最大输出 token\"后重试"
            )
        # 只有真实、非空译文才表示该引擎已恢复；成功后立即解除冷却。
        self._record_success(elapsed_seconds, generation=generation, recovered=True)
        return True, f"连接正常 · {elapsed:.0f}ms · 模型返回：{reply[:40]}"

    # ------------------------------------------------------------------
    # HTTP 层
    # ------------------------------------------------------------------

    def _chat(self, system: str, user: str, max_tokens: int) -> str:
        """发起一次 chat/completions 请求并返回正文内容。"""
        payload = self._build_payload(system, user, max_tokens)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        path = f"{self._prefix}/chat/completions"

        def _send_payload() -> bytes:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            return self._post(path, body, headers)

        try:
            response = _send_payload()
        except ApiRequestError as exc:
            # 400 且错误信息点名了不支持的参数：就地改参数自适应重试。
            # 可学习的怪癖共两处（max_completion_tokens / temperature），
            # 循环上限 2 覆盖"两家怪癖同现"的服务商——OpenAI o 系/gpt-5
            # 既不收 max_tokens 也不收自定义温度，只重试一次的话第二处
            # 要等一次完整冷却期才能学到，首帧白退本地 60 秒。
            # 自适应无效（提示对不上）或两次后仍失败则照常抛出（冷却照走）。
            for _ in range(2):
                if not self._adapt_payload_for_400(exc, payload):
                    raise exc
                try:
                    response = _send_payload()
                    break
                except ApiRequestError as retry_exc:
                    exc = retry_exc
            else:
                raise exc
        data = json.loads(response.decode("utf-8"))
        content = _extract_content(data)
        # 截断判定不依赖正文是否为空：推理类模型把预算全花在思考上时
        # content 为空且 finish_reason=length，翻倍预算同样能救回来。
        if _extract_finish_reason(data) == "length":
            content = self._retry_truncated(system, user, max_tokens, headers, path)
        return content

    def _retry_truncated(
        self,
        system: str,
        user: str,
        max_tokens: int,
        headers: dict[str, str],
        path: str,
    ) -> str:
        """输出被 ``finish_reason="length"`` 截断时，按翻倍预算重问一次。

        :return: 完整译文；翻倍后仍截断（或预算已在硬顶、重试请求失败）
            时返回空串，调用方按失败记账并回退本地模型。重试自身的错误
            **不单独记账**——首次请求是成功的，只有最终拿不到完整译文
            才由调用方的空内容路径记一次失败。
        """
        retry_budget = min(MAX_OUTPUT_TOKENS, max_tokens * 2)
        if retry_budget <= max_tokens:
            logger.warning(
                "在线 AI 输出在预算 %s token 处被截断且已到硬顶，本次回退本地模型",
                max_tokens,
            )
            return ""
        logger.debug(
            "在线 AI 输出被截断（%s token），翻倍预算重问一次", max_tokens
        )
        payload = self._build_payload(system, user, retry_budget)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            response = self._post(path, body, headers)
        except (ApiRequestError, OSError, http.client.HTTPException) as exc:
            logger.warning("在线 AI 截断重试请求失败，本次回退本地模型：%s", exc)
            return ""
        data = json.loads(response.decode("utf-8"))
        content = _extract_content(data)
        if content and _extract_finish_reason(data) != "length":
            return content
        logger.warning(
            "在线 AI 输出翻倍预算（%s token）后仍无完整译文，本次回退本地模型",
            retry_budget,
        )
        return ""

    def _build_payload(self, system: str, user: str, max_tokens: int) -> dict:
        """按已学习的参数兼容标志构建请求体。"""
        payload: dict = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if self._use_completion_tokens:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        if not self._omit_temperature:
            payload["temperature"] = self._temperature
        return payload

    def _adapt_payload_for_400(self, exc: ApiRequestError, payload: dict) -> bool:
        """按 400 错误信息自适应请求参数；需要重试时返回 ``True``。

        就地修改 ``payload``。学习是单向的：一旦学到本服务商的参数
        要求，后续请求直接按正确参数发出，不再触发该 400。
        """
        if exc.status != 400:
            return False
        detail = exc.detail
        if (
            not self._use_completion_tokens
            and _COMPLETION_TOKENS_HINT in detail
            and "max_tokens" in payload
        ):
            self._use_completion_tokens = True
            payload[_COMPLETION_TOKENS_HINT] = payload.pop("max_tokens")
            logger.warning(
                "服务商不接受 max_tokens 参数，已改用 max_completion_tokens 重试"
            )
            return True
        if (
            not self._omit_temperature
            and _TEMPERATURE_HINT in detail
            and "temperature" in payload
        ):
            self._omit_temperature = True
            payload.pop("temperature")
            logger.warning("服务商不接受自定义 temperature，已移除该参数重试")
            return True
        return False

    def _post(self, path: str, body: bytes, headers: dict[str, str]) -> bytes:
        """发送 POST：先复用空闲连接，陈旧连接失败自动换新重试一次。

        只对"复用的旧连接"做一次重试——新建连接上的失败就是真实失败
        （断网、超时），重试只会把等待时间翻倍。
        """
        reused = self._checkout()
        if reused is not None:
            response = self._send(reused, path, body, headers, tolerate_stale=True)
            if response is not None:
                return response
            # 走到这里说明复用的连接已失效（服务端提前关闭等）
        fresh = self._new_connection()
        response = self._send(fresh, path, body, headers, tolerate_stale=False)
        assert response is not None
        return response

    def _send(
        self,
        conn: http.client.HTTPConnection,
        path: str,
        body: bytes,
        headers: dict[str, str],
        tolerate_stale: bool,
    ) -> bytes | None:
        """在给定连接上完成一次请求。

        :return: 响应体；仅当 ``tolerate_stale`` 且连接层失败时返回 ``None``。
        :raises ApiRequestError: 服务商返回非 2xx。
        """
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
        except (OSError, http.client.HTTPException):
            self._discard(conn)
            if tolerate_stale:
                return None
            raise
        if resp.status != 200:
            self._discard(conn)
            raise ApiRequestError(resp.status, _error_detail(data))
        self._checkin(conn)
        return data

    def _new_connection(self) -> http.client.HTTPConnection:
        proxy = _system_proxy_for(self._base_url)
        if proxy is None:
            # 无系统代理：保持原有直连行为
            if self._scheme == "https":
                return http.client.HTTPSConnection(
                    self._host, self._port, timeout=self._timeout,
                    context=self._ssl_context,
                )
            return http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)

        # 系统有代理：通过代理建连（HTTPS 走 CONNECT 隧道；HTTP 直接复用代理连接）
        return _new_connection_through_proxy(
            proxy=proxy,
            target_host=self._host,
            target_port=self._port,
            scheme=self._scheme,
            timeout=self._timeout,
            ssl_context=self._ssl_context,
        )

    def _checkout(self) -> http.client.HTTPConnection | None:
        while True:
            with self._pool_lock:
                conn = (
                    self._idle_connections.pop() if self._idle_connections else None
                )
            if conn is None:
                return None
            if not _connection_dropped(conn):
                return conn
            # 服务端已提前关闭的陈旧连接：直接丢弃换下一条，
            # 免得走"发送失败→丢弃→新建连接"的完整重试往返
            self._discard(conn)

    def _checkin(self, conn: http.client.HTTPConnection) -> None:
        with self._pool_lock:
            if not self._closed and len(self._idle_connections) < 8:
                self._idle_connections.append(conn)
                return
        with contextlib.suppress(Exception):
            conn.close()

    def _discard(self, conn: http.client.HTTPConnection | None) -> None:
        if conn is None:
            return
        with contextlib.suppress(Exception):
            conn.close()

    # ------------------------------------------------------------------
    # 指标与冷却
    # ------------------------------------------------------------------

    def _request_generation(self) -> int:
        with self._metrics_lock:
            return self._recovery_generation

    def _record_success(
        self, latency: float, *, generation: int, recovered: bool = False
    ) -> None:
        with self._metrics_lock:
            self._total_calls += 1
            if generation != self._recovery_generation:
                return
            if recovered:
                self._recovery_generation += 1
                # Invalidate requests/cache keys prepared before recovery,
                # including old requests that will return a local fallback.
                self._failure_generation += 1
            self._last_latency = latency
            self._consecutive_failures = 0
            self._last_error_status = None

    def _record_failure(self, status: int | None = None, *, generation: int) -> None:
        with self._metrics_lock:
            self._total_calls += 1
            if generation != self._recovery_generation:
                return
            self._last_latency = 0.0
            self._failure_generation += 1
            self._last_error_status = status
            self._last_failure_at = time.monotonic()
            if status in DETERMINISTIC_FAILURE_STATUSES:
                # 请求被拒/认证/权限/路径错误重试必然复现：一次即判冷却，
                # 同帧后续片段直接走本地，不再逐段白付往返
                self._consecutive_failures = FAILURE_THRESHOLD
                logger.warning(
                    "在线 AI 返回 HTTP %s（配置或认证类错误），%s 秒内直接回退本地模型",
                    status,
                    COOLDOWN_SECONDS,
                )
                return
            self._consecutive_failures += 1
            if self._consecutive_failures == FAILURE_THRESHOLD:
                logger.warning(
                    "在线 AI 连续 %s 次失败，%s 秒内直接回退本地模型",
                    FAILURE_THRESHOLD,
                    COOLDOWN_SECONDS,
                )

    def _in_cooldown(self) -> bool:
        with self._metrics_lock:
            if self._consecutive_failures < FAILURE_THRESHOLD:
                return False
            if time.monotonic() - self._last_failure_at >= COOLDOWN_SECONDS:
                # 冷却期满：清零计数，重新尝试在线请求
                self._consecutive_failures = 0
                return False
            return True


class ApiRequestError(Exception):
    """服务商返回非 2xx 状态码。"""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def _connection_dropped(conn: http.client.HTTPConnection) -> bool:
    """零开销探测空闲连接是否已被服务端关闭。

    HTTP/1.1 keep-alive 下响应总被读尽，socket 若再变为可读只可能是
    对端发来 EOF（或异常数据）——复用必然失败。用 ``select`` 零超时
    探测在取用前检出，省掉一次失败的发送＋重试往返（含重新握手
    100–300ms）；健康的连接此调用立刻返回 ``False``，无任何系统级等待。
    """
    sock = conn.sock
    if sock is None:
        return False
    try:
        readable, _, _ = select.select([sock], [], [], 0)
    except (OSError, ValueError):
        return True
    return bool(readable)


def _extract_content(data: dict) -> str:
    """从响应 JSON 中取正文；缺失结构按空串处理（失败路径统一兜底）。"""
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        # 部分 OpenAI 兼容网关返回分片数组 [{"type": "text", "text": "..."}]。
        # 按文本分片拼接；不解这个形态会把成功响应误判成空内容，
        # 连续三次即触发冷却，在线模式整段白退本地。
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        joined = "".join(parts)
        if joined:
            return joined
    # 旧版补全接口的响应结构
    text = choices[0].get("text")
    return text if isinstance(text, str) else ""


def _extract_finish_reason(data: dict) -> str:
    """取 ``choices[0].finish_reason``；缺失或类型异常按空串处理。"""
    choices = data.get("choices") or []
    if not choices:
        return ""
    reason = choices[0].get("finish_reason")
    return reason if isinstance(reason, str) else ""


def _error_detail(data: bytes) -> str:
    """尽力从错误响应里提取可读信息（OpenAI 风格 ``error.message``）。"""
    try:
        parsed = json.loads(data.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError):
        return data[:200].decode("utf-8", errors="replace")
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(parsed)[:200]


def _friendly_error(exc: Exception) -> str:
    """把请求异常转成用户能看懂的中文提示（测试连接用）。"""
    if isinstance(exc, ApiRequestError):
        if exc.status == 400:
            return "请求被拒（400）：请检查模型名与参数是否正确"
        if exc.status == 401:
            return "认证失败（401）：API Key 无效或已过期"
        if exc.status == 402:
            return "余额不足（402）：请前往服务商控制台充值或确认套餐配额"
        if exc.status == 403:
            return "无权限（403）：请确认 Key 对该模型有访问权限"
        if exc.status == 404:
            return "接口不存在（404）：请检查接口地址与路径"
        if exc.status == 429:
            return "请求过多（429）：已触发服务商限流，稍后再试"
        if exc.status >= 500:
            return f"服务商服务器错误（{exc.status}）：稍后再试"
        return f"请求失败（HTTP {exc.status}）：{exc.detail[:120]}"
    return f"无法连接：{exc}"[:160]
