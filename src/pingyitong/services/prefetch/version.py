"""模型版本指纹（缓存键第五元）。

模型升级 / 换量化 / 换提示词模板都会使旧译文失效——这个指纹是缓存键的
一部分（见 SentenceCache.cache_key 第五元），任何一项变化 → 键变化 →
旧条目自然过期，不会出现"新模型显示旧译文"。
"""

from __future__ import annotations

import hashlib

from ...components.llm.llama_server import CHAT_TEMPLATE_NAME


def model_version_fingerprint(
    fast_file: str,
    accurate_file: str,
    app_version: str,
    template_version: str,
) -> str:
    """把影响译文的要素压成一个短指纹。

    不直接用版本号拼接原文——指纹短且稳定，日志里好认；
    碰撞概率对"失效判定"场景可以忽略（sha256 截断 16 字节）。
    """
    material = "|".join((fast_file, accurate_file, app_version, template_version))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def fingerprint_for_config(config) -> str:
    """从应用配置构造指纹（供 Application 装配 SentenceCache）。

    模板版本取 chat template 文件内容的哈希前 8 位：模板是影响译文的
    真实变量（GGUF 内不含模板，由本项目随附），内容变更 = 旧译文失效。
    """
    from ...components.llm.llama_server import default_chat_template
    from ...version import version_string

    template = default_chat_template()
    template_version = CHAT_TEMPLATE_NAME
    try:
        if template is not None and template.is_file():
            template_version = (
                CHAT_TEMPLATE_NAME + ":"
                + hashlib.sha256(template.read_bytes()).hexdigest()[:8]
            )
    except OSError:
        pass

    return model_version_fingerprint(
        fast_file=getattr(config.model.fast, "file_name", ""),
        accurate_file=getattr(config.model.accurate, "file_name", ""),
        app_version=version_string(),
        template_version=template_version,
    )
