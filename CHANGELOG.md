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

- `python -m pytest tests/test_urlopen_proxy_safe.py -v`：18 passed。
- `python -m pytest tests/`：1734 passed + 1088 subtests passed（84.65s）。
  较 0.6.40 记录的 1716 + 1088 增加 18 条，与新增测试文件条数一致。
- `python -m ruff check src tests`：All checks passed。

### 修复（验收后补丁）

- **降级分支兑现「抛首次异常」契约**
  - `utils/urlopen_proxy_safe.py` 的 `open_with_proxy_fallback` 原实现直接
    `return direct_opener.open(...)`，降级再次失败时抛的是直连的**第二次**
    异常，与模块 docstring 及本节「两次均失败则抛出首次原始异常」相反。
    后果是 `web_fetcher` 转成 `FetchResult(kind="error")`、`network_monitor`
    吞成 `False` 时，把代理端点故障记账成站点不可达。
  - 修复：降级分支包 `except Exception: raise first from None`。
  - 补 2 条用例覆盖此前完全空缺的「两次均失败」分支：默认路径与透传
    handler 路径各一条；断言异常对象恒为首次、`__suppress_context__` 为真。
  - 该缺陷由 0.6.41 首次打包后的验收发现，原 16 条用例未覆盖该分支，
    故已生成的旧安装包内含旧行为，本次已重新打包覆盖。

### 发布产物（2026-10-23，含验收后重建）

- `python build/build_exe.py --installer`（onedir，未含 CUDA）：
  - `build/dist/PingYiTong/PingYiTong.exe`：14.2 MB，SHA256 `6a19e119…58a76e`。
  - `build/Output/屏译通-Setup-0.6.41.exe`：2.7 MB；文件版本 0.6.41。
  - `build/Output/屏译通-Setup-0.6.41-1.bin`：3.9 GB。
- `scripts/verify_packaged_release.py --version 0.6.41`：`version_matches=true`，
  69 个函数、100 个模块、20 个 OCR 资产与源码逐项一致（`passed=true`）；
  另反汇编包内 `open_with_proxy_fallback` 确认存在 `RAISE_VARARGS(2)`
  （即 `raise first from None`），证明补丁确实进入安装包。
- `scripts/install_verified_release.ps1 -Version 0.6.41`：静默安装 exit 0，
  注册表版本/路径一致，built 与 installed EXE 的 SHA256 同为 `6A19E119…8A76E`，
  用户 `config.json` 前后哈希一致（证据：`artifacts/install_0641_rebuild/`，不入库）。
- 分发包 `屏译通-Setup-0.6.41.zip`（`testzip=None`、成员哈希一致）及
  `parts_0641/*.part` 六卷拼接 SHA256 与 zip 相同（`9bb13176…bec86c`）；
  校验清单 `build/Output/SHA256SUMS-0.6.41.txt` 已随重建重算（9 条）。
- 重建后产物 SHA256：
  - `屏译通-Setup-0.6.41.exe` = `b5a7afa3fe95e01e6f9f9876004b66f68c8f0e9ce1a6c09777e39c39c74b2756`
  - `屏译通-Setup-0.6.41-1.bin` = `88236eca6703ada3f565899c73991489ac3392cc9de41fd829e1945096ec47af`

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
