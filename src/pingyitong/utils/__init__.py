"""通用工具：路径解析、日志、单实例控制、平台信息。"""

from __future__ import annotations

from .exceptions import install_exception_hooks, report_count
from .logger import describe_text, get_logger, setup_logging
from .paths import app_data_dir, asset_path, icon_path, project_root
from .process_guard import assign_to_job, reap_stale_servers
from .single_instance import SingleInstance

__all__ = [
    "SingleInstance",
    "app_data_dir",
    "asset_path",
    "assign_to_job",
    "describe_text",
    "get_logger",
    "icon_path",
    "install_exception_hooks",
    "project_root",
    "reap_stale_servers",
    "report_count",
    "setup_logging",
]
