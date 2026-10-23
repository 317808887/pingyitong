"""版本信息。

版本号遵循《产品功能规划》五、版本规划中的阶段划分：
V0.1 原型验证 / V0.5 核心功能 / V0.9 体验优化 / V1.0 正式发布。
"""

from __future__ import annotations

from typing import Final

__version__: Final[str] = "0.6.41"
__stage__: Final[str] = "核心功能"

#: 与版本对应的阶段说明，用于关于对话框中展示。
VERSION_STAGE_NOTES: Final[dict[str, str]] = {
    "0.1": "原型验证：屏幕捕获 + OCR + 翻译 + 基础渲染最小原型",
    "0.5": "核心功能：双模型调度、原位渲染优化、主 UI 完成",
    "0.9": "体验优化：多屏适配、术语库、历史记录、性能压测",
    "1.0": "正式发布：一键安装包、使用文档、内测发布",
}


def version_string() -> str:
    """返回形如 ``0.5.0 (核心功能)`` 的完整版本描述。"""
    return f"{__version__} ({__stage__})"
