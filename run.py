#!/usr/bin/env python3
"""Unified one-click launcher for the research multi-agent system.

Starts:
- Master Multi-Agent Brain (LangGraph, ChromaDB, BGE-M3, BGE-Reranker)
- Whisper Live Voice & Audio Stream Worker (WebSocket)
- Web research assistant frontend (http://localhost:8008)
- MCP Server Compatibility
"""

import sys
from pathlib import Path

# 添加当前目录到环境变量
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    from backend_api import run_server
    run_server()
