# 变更记录

版本号格式遵循 `pyproject.toml` / `src/pingyitong/version.py` 的单一来源（`build/installer.iss`
由 `build/build_exe.py` 从 `version.py` 读取注入，不单独维护）。

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
