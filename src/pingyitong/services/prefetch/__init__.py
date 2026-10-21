"""联网预翻译子包。

方案见 docs/联网预翻译技术方案.md（v1.1）。模块清单与职责：

- :mod:`.normalizer` —— 归一化与分句（OCR 通道与字幕行共用）
- :mod:`.sentence_cache` —— 持久化句级缓存（SQLite/WAL/LRU/fuzzy）
- :mod:`.network_monitor` —— 可达性探测 + 计费判定
- :mod:`.gate_state` —— 双层状态机（全局网关 + 每源健康）
- :mod:`.lookup` —— 接缝 A：翻译服务依赖的查询协议与默认实现

调度器 / 字幕源 / 网页抓取 / 对照视图属 P2、P3，见方案实施计划。
"""

from __future__ import annotations

from .active_track import ActiveTrackRegistry, SubtitleTrack, next_episode_file
from .econ_gate import EconGate, SiteStats, threshold
from .gate_state import GateSnapshot, GateState, PrefetchGate, SourceHealth
from .lookup import LookupContext, PrefetchLookup, SentenceCacheLookup
from .metrics import MetricsSnapshot, PrefetchMetrics
from .network_monitor import NetState, NetworkCost, NetworkMonitor
from .reader_view import ReaderView
from .scheduler import PrefetchJob, PrefetchScheduler, Priority
from .sentence_cache import FuzzyHit, SentenceCache, SentenceEntry
from .subtitle_source import SubtitleLine, parse_subtitle_file

__all__ = [
    "ActiveTrackRegistry",
    "EconGate",
    "FuzzyHit",
    "GateSnapshot",
    "GateState",
    "LookupContext",
    "MetricsSnapshot",
    "NetState",
    "NetworkCost",
    "NetworkMonitor",
    "PrefetchGate",
    "PrefetchJob",
    "PrefetchLookup",
    "PrefetchMetrics",
    "PrefetchScheduler",
    "Priority",
    "ReaderView",
    "SentenceCache",
    "SentenceCacheLookup",
    "SentenceEntry",
    "SiteStats",
    "SubtitleLine",
    "SubtitleTrack",
    "SourceHealth",
    "next_episode_file",
    "parse_subtitle_file",
    "threshold",
]
