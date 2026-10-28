"""Unified System Configuration and Hardware/Environment Decoupling Utilities.

Eliminates all developer machine-specific hardcoded paths, hardware-bound audio sinks,
and inconsistencies across LLM models and timeouts.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def get_default_llm_model(default: str = "gpt-5.6-terra") -> str:
    """统一解析当前激活的大模型名称，消除环境变量分歧与硬编码默认值"""
    return os.getenv("DEFAULT_LLM_MODEL") or os.getenv("OPENAI_MODEL", default)


def get_llm_timeout(default: float = 100.0) -> float:
    """统一解析全局大模型请求超时时间 (秒)"""
    val = os.getenv("LLM_TIMEOUT")
    if val:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return float(default)


def resolve_model_path(
    env_key: str,
    local_subpath: str,
    fallback_hf_id: str,
    base_dir: Optional[Path] = None,
) -> str:
    """通用本地模型路径解析器。
    
    优先级：
    1. 环境变量指定的路径（支持相对路径如 ./models/... 或 ../models/...，自动基于 base_dir 解析）
    2. 项目内相对路径 <base_dir>/models/<local_subpath>
    3. 同级相对路径 <base_dir>/../models/<local_subpath>
    4. 标准 HuggingFace 模型标识符 (fallback_hf_id)
    严禁任何特定宿主机绝对路径硬编码。
    """
    val = os.getenv(env_key)
    if val:
        p = Path(val)
        if not p.is_absolute() and base_dir:
            rel_candidate = (base_dir / p).resolve()
            if rel_candidate.exists():
                return str(rel_candidate)
        if p.exists():
            return str(p.resolve())
        if not val.startswith("/") and not val.startswith("."):
            return val

    if base_dir:
        # 1. 项目内 models 目录
        local_dir = (base_dir / "models" / local_subpath).resolve()
        if local_dir.exists():
            return str(local_dir)
        # 2. 项目同级 models 共享资产目录
        peer_dir = (base_dir.parent / "models" / local_subpath).resolve()
        if peer_dir.exists():
            return str(peer_dir)

    return val or fallback_hf_id


def resolve_whisper_path(base_dir: Optional[Path] = None) -> Path:
    """解析 Whisper 语音模型路径，全动态相对路径支持与零硬编码"""
    val = os.getenv("WHISPER_MODEL_PATH")
    if val:
        p = Path(val)
        if not p.is_absolute() and base_dir:
            rel_candidate = (base_dir / p).resolve()
            if rel_candidate.exists():
                return rel_candidate
        if p.exists():
            return p.resolve()
        if not val.startswith("/") and not val.startswith("."):
            return Path(val)

    if base_dir:
        local_dir = (base_dir / "models" / "whisper-large-v3").resolve()
        if local_dir.exists():
            return local_dir
        peer_dir = (base_dir.parent / "models" / "whisper-large-v3").resolve()
        if peer_dir.exists():
            return peer_dir

    return Path(val or "openai/whisper-large-v3")


def get_default_monitor_source() -> str:
    """动态探测系统活动的扬声器内录 (Speaker Monitor) 音频源。
    
    解耦对特定主板 PCI 声卡硬件设备名的写死。
    优先级：
    1. 环境变量 AUDIO_MONITOR_SOURCE (若用户显式指定)
    2. 通过 pactl 动态探测 Speaker monitor 或任意 monitor
    3. 缺省安全标识符 'default'
    """
    env_source = os.getenv("AUDIO_MONITOR_SOURCE")
    if env_source:
        return env_source

    try:
        res = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
        # 优先匹配 Speaker monitor
        for line in res.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and "monitor" in parts[1] and "Speaker" in parts[1]:
                return parts[1]
        # 次优匹配任意 monitor
        for line in res.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and "monitor" in parts[1]:
                return parts[1]
    except Exception:
        pass

    return "default"


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
