# 项目工作状态存档 — 2026-10-21

本文档由 Crow5 会话生成，记录截至当日的推进状态，证据均来自本会话实际执行结果。

## 当前做到哪一步

- 项目：屏译通（PingYiTong），基于本地大模型的 Windows 桌面实时原位翻译系统。
- 形态：Python / PyQt6 单体桌面应用，无独立的前后端服务代码；"前端 + 后端"交付物即本仓库全部源码（src/pingyitong、scripts、tests、config、docs）。
- 版本：0.6.39（src/pingyitong/version.py，据分析轮读取）。
- 本会话主线目标「上传 GitHub 并开源」的主体动作已完成，详见下节。

## 已完成（本会话，有实证）

- 本地 git 仓库初始化于 `F:\讯飞\项目源码`（main 分支），首次提交 `612ba8a`：290 个文件、97,420 行，覆盖全部源码、测试、配置与文档。
- `.gitignore` 扩展：新增排除 `artifacts/`、`research/`、`managed_iterations/`、`.opencode/`、构建日志与调试截图产物；模型权重（*.gguf）、`runtime/`、数据库（*.db、*.db-wal、*.db-shm）、`data/` 延续原有排除策略。
- 提交前凭据扫描（gho_/ghp_/sk-/xox-/AKIA 模式，暂存区全量）：无匹配。
- GitHub 公开仓库建仓并推送成功：origin = https://github.com/317808887/pingyitong（PUBLIC），`main -> origin/main` 推送成功并建立跟踪；账号 317808887（device-code 授权；所有 GitHub 访问经本机代理 127.0.0.1:7892 完成，直连实测不通）。
- 数据库备份（应用未运行：进程计数 0 的情况下执行）：
  - 库文件集合 6 个：`history.db`（2,523,136 B）、`translation_cache.db`（561,152 B）、`translation_cache.db-wal`（0 B）、`translation_cache.db-shm`（32,768 B）、`prefetch_cache.db`（45,056 B）、`prefetch_metrics.db`（12,288 B）；
  - 源路径 `%LOCALAPPDATA%\PingYiTong\`（data 子目录 + 根级两库），目标路径本仓库 `database\backup_20261021\`；
  - 完整性校验：4 个 .db 全部 `PRAGMA integrity_check = ok`（对象数：history 4、prefetch_cache 8、prefetch_metrics 2、translation_cache 2，脚本退出码 0）；
  - SHA256 清单：`database\backup_20261021\SHA256SUMS.txt`。

## 不入库说明

- 数据库备份文件按 `.gitignore` 既有策略（*.db 等）保留在本地，不推送 GitHub；history.db 含屏幕识别内容，属隐私敏感数据，入网边界不做调整。
- 本仓库不含模型权重与 llama-server 运行时（约 3.45 GB + 600 MB），克隆者按 README 使用 `scripts/download_models.py`、`scripts/download_runtime.py` 获取。

## 待验证

- 0.6.39 全量 pytest / ruff：据 artifacts/continuation_20261019 记录曾为 1423 passed + 1084 subtests、ruff 通过；本会话未重跑，基线是否仍绿需本地复跑确认。
- 版本 0.6.37–0.6.39 的变更说明在 docs 中缺位（分析轮结论），溯源材料在 artifacts/ 内，需补 docs 记录。
- 语义审查 A/B 研究（relation_audit 臂）的同判映射复核与生产提示词切换决策未完成（分析轮结论，见 managed_iterations r0011 进行中的清单）。
- 本文档提交后远端 main HEAD 与本地是否一致，随提交推送一并核对。

## 下一步建议

1. 本地复跑全量 pytest + ruff（对照既有基线），有差异优先排查 0.6.37–0.6.39 涉及的组件（slot_layout / pipeline / overlay_window / ocr 等近期改动文件）。
2. 补 0.6.37–0.6.39 变更记录（docs/ 下带日期文档）。
3. 完成 A/B 同判映射复核后，再决定是否切换生产语义提示词（谨慎：relation_audit 臂以增加误扣换取减少漏放，非免费收益）。
4. 数据库备份流程固化：应用关闭后复制 + integrity_check + SHA256 三步走；必要时写成脚本纳入 scripts/。
5. 后续按功能块做增量提交，避免历史再次停在单条 initial commit 的粗粒度上。
