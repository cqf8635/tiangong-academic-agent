"""Unified System Configuration and Hardware/Environment Decoupling Utilities.

Eliminates all developer machine-specific hardcoded paths, hardware-bound audio sinks,
and inconsistencies across LLM models and timeouts.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def get_default_llm_model() -> str:
    """统一解析当前激活的大模型名称，消除环境变量分歧与硬编码默认值"""
    model = os.getenv("DEFAULT_LLM_MODEL") or os.getenv("OPENAI_MODEL")
    if not model or not model.strip():
        raise ValueError("未配置 DEFAULT_LLM_MODEL 或 OPENAI_MODEL")
    return model.strip()


def get_llm_timeout(default: float = 100.0) -> float:
    """统一解析全局大模型请求超时时间 (秒)"""
    val = os.getenv("LLM_TIMEOUT")
    if val:
        try:
            return float(val)
        except (ValueError, TypeError) as e:
            raise ValueError("LLM_TIMEOUT 配置无效") from e
    return float(default)


def resolve_model_path(env_key: str, local_subpath: str, base_dir: Optional[Path] = None) -> str:
    """使用明确配置或项目内约定的本地目录，不自动改用远程模型。"""
    value = os.getenv(env_key)
    path = Path(value).expanduser() if value else Path("models") / local_subpath
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    if value and not path.exists():
        raise FileNotFoundError(f"{env_key} 指定的模型不存在: {path}")
    return str(path.resolve())


def resolve_whisper_path(base_dir: Optional[Path] = None) -> Path:
    """使用指定的 Whisper 路径；未配置时使用项目内约定目录。"""
    return Path(resolve_model_path("WHISPER_MODEL_PATH", "whisper-large-v3", base_dir))


def get_default_monitor_source() -> str:
    """使用显式音源或当前默认输出设备的内录通道；探测失败直接报错。"""
    source = os.getenv("AUDIO_MONITOR_SOURCE")
    if source:
        return source
    result = subprocess.run(["pactl", "get-default-sink"], capture_output=True,
                            text=True, check=True, timeout=2.0)
    sink = result.stdout.strip()
    if not sink:
        raise RuntimeError("未找到默认输出音源，请配置 AUDIO_MONITOR_SOURCE")
    return f"{sink}.monitor"


def get_default_obsidian_vault_path() -> Path:
    """跨平台动态解析 Obsidian 默认知识库存储路径"""
    custom = os.getenv("OBSIDIAN_VAULT_PATH")
    if custom:
        return Path(custom)

    home = Path.home()
    # 适配中文与英文系统主流文档目录
    doc_en = home / "Documents"
    doc_zh = home / "文档"
    if doc_zh.exists():
        return doc_zh / "Obsidian Vault"
    if doc_en.exists():
        return doc_en / "Obsidian Vault"
    return home / "Obsidian Vault"
