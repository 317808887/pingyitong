# 翻译方向与外部 AI 接入修复实测

日期：2026-09-30；版本：0.6.10；Windows，本地源码与打包程序分别验证。

## 修复结果

- 下拉框同步规范化后的语言对；切换语言/API 后清除旧输出并重新处理静止画面。
- 捕获、OCR、翻译、渲染按代际过滤旧帧；清队列时保护新帧的差分基线。
- 每次请求固定配置、术语、在线模式与路由；过期请求不提交缓存、复制输出或历史。
- API 与本地缓存隔离，在线模式绕过本地预翻译缓存；新 Key/模型立即生效。
- 连接测试复用当前引擎，成功解除冷却并重新翻译；迟到旧失败不能重新冷却。
- 连接测试结果绑定被测配置；现有徽标显示真实在线/本地回退状态。UI 设计不变。

## 最终源码验证

| 检查 | 结果 | 原始记录 |
| --- | --- | --- |
| 完整 unittest | 850 项，65.123 秒，OK，退出码 0 | `artifacts/unittest_routing_fix_20260930.log` |
| 修改文件 Ruff | 通过 | 工具直接输出 |
| 静态导入检查 | 126 个模块通过 | `scripts/check_imports.py` |
| 三方向，本地 1.7B | 方向/批量对位通过；稳态 78/125/94ms | `artifacts/directions_fast_routing_fix_20260930.log` |
| 三方向，本地 4B | 方向/批量对位通过；稳态 125/172/141ms | `artifacts/directions_accurate_routing_fix_20260930.log` |
| 在线路径与真实本地回退 | 单段、12/12 批量对位、连接复用、不可达端口回退通过 | `artifacts/api_routing_integration_20260930.log` |
| 文字栏读取顺序 | 所有测试版面通过 | `artifacts/transcript_order_routing_fix_20260930.log` |
| 源码 self-check | 退出码 0，真实屏幕捕获与 OCR 推理通过 | `artifacts/selfcheck_source_routing_fix_20260930.log` |

稳态耗时只包含脚本的翻译调用，不包含屏幕捕获/OCR/渲染，也不是端到端 p95。
上述方向顺序为英→中、中→英、自动→中文。

## 保存的外部 Key 实际请求

从用户配置读取已启用的 `api.deepseek.com / deepseek-chat`，仅在内存修改
测试语言方向，没有保存测试设置。Key 未打印、未写入测量报告。
连接测试成功后通过产品 ModelPool 与 TranslationService 实际翻译：

| 方向 | 原文 | 译文 | 请求数 | 翻译调用耗时 |
| --- | --- | --- | --- | --- |
| 英→中 | The door is open. | 门开着。 | 1 | 435.1ms |
| 中→英 | 请保存文件，然后关闭窗口。 | Save the file, then close the window. | 1 | 658.5ms |
| 自动→中文 | Click the Settings button. | 点击设置按钮。 | 1 | 549.4ms |

三项均检查 API 调用计数实际增加、失败代际未改变、本地引擎未创建。
各自重复一次相同文本，译文一致，API 调用计数不再增加，证明成功在线缓存复用。
原始记录：`artifacts/api_configured_route_20260930.json`。

## 准确度边界

三条真实 API 样例人工核对语义正确；本地 4B 的三方向样例也语义正确。
本地 1.7B 本轮将 “brown fox” 输出为“棕熊”，这是已观察到的模型词义错误。
方向脚本只自动检查目标语言与批量对位，其退出码 0 不能证明每个词都准确。
本次修复不把该错误报成准确翻译，也不声称任意文本的翻译准确度或在线延迟全部达标。

## 打包与安装

- 最终 `python -X utf8 build/build_exe.py --installer --no-clean` 退出码 0。
- 安装包为 `build/Output/屏译通-Setup-0.6.10.exe` 与
  `build/Output/屏译通-Setup-0.6.10-1.bin`；两者需放在同一目录。
  数据分卷为 3,839,463,936 字节（约 3.84GB）。
- 已独立检查 exe 内嵌的请求上下文、缓存路由、连接恢复、输出清除与运行状态
  代码，包含最后的状态徽标修复；记录：`artifacts/packaged_routing_modules_20260930.json`。
- 本机静默覆盖安装退出码 0，注册版本为 0.6.10。安装路径：
  `C:/Users/31780/AppData/Local/Programs/屏译通/PingYiTong.exe`。
  安装后的 exe SHA256 与最终构建产物相同，用户配置文件 SHA256 前后相同，
  配置与 Key 保留。记录：`artifacts/install_routing_fix_20260930.json`。
- 安装版 `--self-check` 生成了本轮的新报告：两屏实际捕获、真实 OCR
  “Self test 12345”、两份模型、安装目录推理运行时及数据目录均通过。
  报告：`artifacts/selfcheck_installed_routing_fix_20260930.log`。
  无控制台版在报告完成后等待自检弹窗，本轮读取报告后清理该测试进程；
  **不把这个进程记作正常退出码 0**。源码自检的退出码 0 已另行确认。

修改源码的 SHA256 存档：`artifacts/routing_fix_source_sha256_20260930.json`。
