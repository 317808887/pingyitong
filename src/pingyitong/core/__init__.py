"""核心调度层。

对应《项目全栈开发文档》一·（三），负责协调各基础组件工作：

- :class:`~pingyitong.core.pipeline.Pipeline` 四级异步流水线调度
- :class:`~pingyitong.core.model_scheduler.ModelScheduler` 双模型动态调度
- :class:`~pingyitong.core.resource_monitor.ResourceMonitor` 系统资源监控
- :class:`~pingyitong.core.coordinate.CoordinateMapper` 坐标映射对齐
- :mod:`~pingyitong.core.task_queue` 队列与背压策略
"""

from __future__ import annotations

from .coordinate import CoordinateMapper, MappedBox
from .model_scheduler import ModelScheduler, ScheduleDecision
from .pipeline import Pipeline, PipelineStatus
from .resource_monitor import ResourceMonitor, ResourceSnapshot
from .task_queue import LatestQueue, QueueStats, RateLimiter

__all__ = [
    "CoordinateMapper",
    "LatestQueue",
    "MappedBox",
    "ModelScheduler",
    "Pipeline",
    "PipelineStatus",
    "QueueStats",
    "RateLimiter",
    "ResourceMonitor",
    "ResourceSnapshot",
    "ScheduleDecision",
]
