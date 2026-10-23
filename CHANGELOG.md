# 变更记录

版本号格式遵循 `pyproject.toml` / `src/pingyitong/version.py` 的单一来源（`build/installer.iss`
由 `build/build_exe.py` 从 `version.py` 读取注入，不单独维护）。

## 0.6.41（2026-10-23）

### 新增

- **`utils/urlopen_proxy_safe.py`：本机代理端点故障时降级直连**
  - `open_with_proxy_fallback(request, timeout, *, handlers=())` 为通用核心，
    支持透传额外 handler；`urlopen_with_proxy_fallback()` 为无额外 handler 版本。
  - `local_proxy_endpoint_unreachable(exc)` 公开判据：系统代理全部为 loopback
    （127.0.0.1 / localhost / ::1）且异常为连接被拒 / 超时形态时返回真；
    未配置代理、远程代理、HTTP 状态码、DNS 失败均不降级。
  - 首次请求仍走默认 opener（跟随系统代理），仅在命中判据时用
    `ProxyHandler({})` 直连重试一次；两次均失败则抛出**首次**原始异常。

### 修复

- **`web_fetcher` 与 `network_monitor` 接入代理降级**
  - `web_fetcher.robots_allows` 与 `fetch` 正文路径改用
    `open_with_proxy_fallback`；新增 `_open_noredirect()` 透传 `_NoRedirect`，
    使首次与降级直连两条路径都保持「手动逐跳重定向」语义。
  - `network_monitor._probe_http` 改用 `urlopen_with_proxy_fallback`。
  - 附注：本机系统代理（fcclient 等）抖动时，原实现会把代理端点自身故障
    记为站点 `timeout`，连续 3 次后该站进入 COOLDOWN。

- **最小字号下限 9px → 12px**
  - `config/schema.py`：`StyleConfig.min_font_size` 默认值 12。
  - `components/render/layout.py`、`components/render/slot_layout.py`：
    `getattr(style, "min_font_size", 9)` 兜底值改为 12，共 3 处。
  - `config/settings.py`：`min_font_size` 与 `font_size_fixed` 的规范化下限
    由 6 提升到 12。

### 测试与构建

- `python -m pytest tests/test_urlopen_proxy_safe.py -v`：16 passed。
- `python -m pytest tests/`：1732 passed + 1088 subtests passed（85.55s）。
  较 0.6.40 记录的 1716 + 1088 增加 16 条，与新增测试文件条数一致。
- `python -m ruff check src tests`：All checks passed。

## 0.6.40（2026-10-22）

### 修复

- **删除 `web_fetcher.fetch` 中不可达的重定向分支**
  - 位置：`src/pingyitong/services/prefetch/web_fetcher.py`，`fetch()` 内 `except HTTPError`
    之后的 `if response.status in (301, 302, 303, 307, 308):` 代码块（约 157–168 行）。
  - 根因：`_NoRedirect` 的 `redirect_request` 返回 `None`，urllib 会把 3xx 抛成
    `HTTPError`，因此 `opener.open()` 成功返回时 `response.status` 恒不为 3xx，该分支不可达。
  - 修复：删除该死分支；重定向逐跳校验统一由 `except HTTPError` 分支承担。
  - 附带影响：原分支中 `if validate:` 门控（与 `except` 分支的无条件校验不一致）随之消失，
    两条路径行为不再有差异。
  - 验证：`tests/test_prefetch_p3.py` 新增 2 条回归用例
    （`test_redirect_to_loopback_blocked_even_without_validate`、
    `test_redirect_missing_location_error`），`/loop` 302 跳私网、
    `/self_redirect` 302 跳回 loopback（validate=False）、`/noloc` 302 无 Location
    三条路径均按预期返回 blocked/error；原有
    `test_redirect_to_private_blocked` 保持通过。

### 测试与构建

- `python -m pytest tests/`：1716 passed + 1088 subtests（较 0.6.39 的 1714 + 1088 增加 2 条）。
- `ruff check src tests`：All checks passed。
- 打包：`python build/build_exe.py --installer`
  （PyInstaller 6.22.3 onedir + Inno Setup 6），产物
  `build/Output/屏译通-Setup-0.6.40.exe`（2,783,936 字节）
  与 `build/Output/屏译通-Setup-0.6.40-1.bin`（3,931,828,820 字节）。
- SHA256：
  - `屏译通-Setup-0.6.40.exe` = `d4839a3596e69bca5d476c1ff7849cec3e2ef46bf5ae0fe02518ee763434bf1a`
  - `屏译通-Setup-0.6.40-1.bin` = `4a9f4926dfcb200d4f36b23dc43932d37347c23ffd25a2e95aca70b3efaddaca`
  - 合订 `PingYiTong-Setup-0.6.40.zip` = `6253c58ce50d2e95582e6471e51462929538b7f681e6443eaeaa739e09c76fa3`
    （3 分卷拼接后逐字节校验一致）

## 历史版本

0.6.37–0.6.39 的变更说明见 `docs/变更记录_0.6.37-0.6.39_20261021.md`。
