"""屏译通 —— 基于本地大模型的桌面实时原位翻译系统。

本包按《项目全栈开发文档》一、系统总体架构划分为五层：

``components``
    基础组件层：屏幕捕获、OCR 识别、图像处理、大模型推理、窗口渲染。
``core``
    核心调度层：流水线调度、双模型调度、资源监控、坐标映射。
``services``
    业务功能层：实时翻译、区域翻译、热键管理、术语库、翻译历史。
``ui``
    UI 交互层：系统托盘、主设置面板、悬浮控制条、原位翻译透明层。
``config`` / ``utils``
    配置持久化与通用工具，为上述各层提供支撑。
"""

from __future__ import annotations

from .version import __stage__, __version__, version_string

__all__ = ["__version__", "__stage__", "version_string"]
