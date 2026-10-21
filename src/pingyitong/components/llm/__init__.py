"""大模型推理组件。

- :class:`~pingyitong.components.llm.model_pool.ModelPool` 双模型池
- :class:`~pingyitong.components.llm.llama_engine.LlamaEngine` llama.cpp 引擎
- :class:`~pingyitong.components.llm.api_engine.ApiEngine` 在线 AI 引擎（OpenAI 兼容）
- :mod:`~pingyitong.components.llm.prompt` 翻译 Prompt 工程
- :class:`~pingyitong.components.llm.translation_cache.TranslationCache` 推理缓存
- :class:`~pingyitong.components.llm.cache_store.PersistentCacheStore` 缓存持久层（SQLite）
"""

from __future__ import annotations

from .api_engine import ApiEngine, ApiRequestError
from .backend import BackendKind, BackendProbe, probe
from .base import (
    EngineMetrics,
    EngineStatus,
    LayoutPreviewEngine,
    TranslationEngine,
)
from .cache_store import PersistentCacheStore
from .llama_engine import LlamaEngine, llama_cpp_available
from .llama_server import LlamaServerEngine, kill_orphans
from .model_pool import ModelPool, PreloadProgress
from .prompt import (
    PromptPair,
    build_prompt,
    build_system_prompt,
    build_user_prompt,
    language_name,
    parse_response,
)
from .translation_cache import CacheStats, TranslationCache

__all__ = [
    "ApiEngine",
    "ApiRequestError",
    "BackendKind",
    "BackendProbe",
    "CacheStats",
    "EngineMetrics",
    "EngineStatus",
    "LayoutPreviewEngine",
    "LlamaEngine",
    "LlamaServerEngine",
    "ModelPool",
    "PersistentCacheStore",
    "PreloadProgress",
    "PromptPair",
    "TranslationCache",
    "TranslationEngine",
    "build_prompt",
    "build_system_prompt",
    "build_user_prompt",
    "kill_orphans",
    "language_name",
    "llama_cpp_available",
    "parse_response",
    "probe",
]
