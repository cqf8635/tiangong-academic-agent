"""FastAPI backend for the research multi-agent workflow and web frontend.

Provides:
- REST API for Chat, Document Ingest, Meeting Summaries, Audio Stream.
- WebSocket for live VAD audio events & status broadcasting.
- Static file serving for the research assistant web UI.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("uvicorn.error")

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from document_rag_engine import UniversalDocumentRAGEngine
from intent_recognition_agent import IntentRecognitionAgent
from master_agent_workflow import GLOBAL_MEMORY_SAVER, MasterAgentState, build_master_workflow, register_stream_callback, system_control_executor
from system_controls import execute_system_control
from meeting_summarizer import MeetingSummarizer
from online_paper_search_agent import OnlinePaperSearchAgent

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

app = FastAPI(title="Research Multi-Agent System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
RECORDS_DIR = BASE_DIR / "records"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

# 懒加载单例组件 (共享全局单例，避免重复初始化)
rag_engine = UniversalDocumentRAGEngine()
meeting_summarizer = MeetingSummarizer()
paper_searcher = OnlinePaperSearchAgent()
from radar_sync_engine import RadarSyncEngine
radar_sync_engine = RadarSyncEngine(paper_searcher)

from master_agent_workflow import MasterEngineRegistry
MasterEngineRegistry.rag_engine = rag_engine
MasterEngineRegistry.meeting_summarizer = meeting_summarizer
MasterEngineRegistry.paper_search_agent = paper_searcher
master_app = build_master_workflow()


@app.on_event("startup")
async def on_app_startup():
    """应用启动挂钩：自动拉起天工雷达后台静默轮询与断点续航同步引擎，并预热音频大模型"""
    radar_sync_engine.start_scheduler()
    voice_manager.event_loop = asyncio.get_running_loop()

    # 预热本地 Whisper-large-v3，默认优先 CUDA；不可用或实际显存不足才回退 CPU。
    def warmup_whisper():
        try:
            from voice_langgraph_agent import WhisperEngine
            print("🚀 [Startup Warmup] 正在预加载本地 Whisper-large-v3 音频模型...")
            backend = WhisperEngine.warmup()
            print(f"✅ [Startup Warmup] 本地 Whisper-large-v3 音频模型预热完成 ({backend})，随时待命！")
        except Exception as e:
            print(f"⚠️ [Startup Warmup] Whisper 模型预热提示: {e}")

    import threading
    threading.Thread(target=warmup_whisper, daemon=True).start()


@app.on_event("shutdown")
async def on_app_shutdown():
    radar_sync_engine.stop_scheduler()


from session_manager import ChatMessage, ConversationSession, SessionManager

# ==========================================
# 1. 领域会话频道管理 (Domain Workspaces & Sessions)
# ==========================================
class CreateSessionRequest(BaseModel):
    title: str
    domain: str = "general"


from config_utils import get_default_llm_model, get_llm_timeout

@app.get("/api/config")
async def get_system_config():
    """获取系统运行期动态配置（如当前激活的大模型名称）"""
    return {
        "model_name": get_default_llm_model(),
        "llm_timeout": get_llm_timeout(100.0),
    }


# ==========================================
# 1.1 系统 .env 可视化配置与本地模型一键下载 (仅限本机访问)
# ==========================================
_SETTINGS_ALLOWED_KEYS = {
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_FALLBACK_MODEL",
    "TRANSLATION_LLM_MODEL",
    "TRANSLATION_BASE_URL",
    "TRANSLATION_API_KEY",
    "LLM_TIMEOUT",
    "OPENALEX_MAILTO",
    "SEMANTICSCHOLAR_API_KEY",
    "HF_ENDPOINT",
    "PAPER_EMBEDDING_MODEL",
    "PAPER_RERANKER_MODEL",
    "PAPER_USE_RERANKER",
    "WHISPER_MODEL_PATH",
}
_SETTINGS_SECRET_KEYS = {"OPENAI_API_KEY", "TRANSLATION_API_KEY", "SEMANTICSCHOLAR_API_KEY"}
_SETTINGS_RESTART_KEYS = {"PAPER_EMBEDDING_MODEL", "PAPER_RERANKER_MODEL", "PAPER_USE_RERANKER", "WHISPER_MODEL_PATH"}

_MODEL_DOWNLOAD_LOCK = threading.Lock()
_MODEL_DOWNLOAD_STATE: Dict[str, Any] = {
    "running": False,
    "target": "",
    "status": "idle",
    "returncode": None,
    "logs": [],
}


def _require_localhost(request: Request) -> None:
    host = (request.client.host if request.client else "") or ""
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="系统配置与模型下载仅允许本机 (localhost) 访问")


def _mask_secret(val: str) -> str:
    s = (val or "").strip()
    if not s:
        return ""
    if len(s) <= 8:
        return "*" * len(s)
    return f"{s[:4]}****{s[-4:]}"


def _read_env_file_map(env_path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not env_path.exists():
        return data
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            data[k] = v
    return data


def _write_env_file_updates(env_path: Path, updates: Dict[str, str]) -> None:
    lines = env_path.read_text(encoding="utf-8", errors="ignore").splitlines() if env_path.exists() else []
    remaining = dict(updates)
    new_lines: List[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        matched_key = None
        if stripped and "=" in stripped:
            cand = stripped.lstrip("#").strip().split("=", 1)[0].strip()
            if cand in remaining and (not stripped.startswith("#") or cand not in {l.strip().split("=", 1)[0].strip() for l in lines if l.strip() and not l.strip().startswith("#") and "=" in l}):
                matched_key = cand
        if matched_key is not None:
            val = remaining.pop(matched_key)
            new_lines.append(f"{matched_key}={val}")
        else:
            new_lines.append(raw_line)
    for k, v in remaining.items():
        new_lines.append(f"{k}={v}")
    env_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


def _check_local_models_status() -> Dict[str, Dict[str, Any]]:
    from download_vector_models import resolve_model_root
    model_root = resolve_model_root()
    env_map = _read_env_file_map(ENV_PATH)

    def _resolve_dir(env_key: str, default_sub: str) -> Path:
        raw = (env_map.get(env_key) or os.getenv(env_key) or "").strip()
        if raw:
            p = Path(raw)
            return p if p.is_absolute() else (BASE_DIR / p).resolve()
        return (model_root / default_sub).resolve()

    emb_dir = _resolve_dir("PAPER_EMBEDDING_MODEL", "embedding")
    rerank_dir = _resolve_dir("PAPER_RERANKER_MODEL", "reranker")
    whisper_dir = _resolve_dir("WHISPER_MODEL_PATH", "whisper")

    def _dir_has_weights(p: Path) -> bool:
        if not p.exists():
            return False
        if p.is_file():
            return p.stat().st_size > 10 * 1024 * 1024
        for ext in ("*.safetensors", "*.bin", "*.pt", "config.json"):
            if any(p.glob(ext)):
                return True
        return False

    whisper_ready = _dir_has_weights(whisper_dir) or (model_root / "whisper" / "large-v3.pt").exists() or (BASE_DIR / "models" / "whisper-large-v3").exists()
    return {
        "embedding": {
            "label": "BGE-M3 向量嵌入模型",
            "repo": "BAAI/bge-m3",
            "path": str(emb_dir),
            "ready": _dir_has_weights(emb_dir),
        },
        "reranker": {
            "label": "BGE-Reranker 神经重排模型",
            "repo": "BAAI/bge-reranker-v2-m3",
            "path": str(rerank_dir),
            "ready": _dir_has_weights(rerank_dir),
        },
        "whisper": {
            "label": "Whisper-large-v3 离线语音模型",
            "repo": "openai/whisper-large-v3 (large-v3.pt)",
            "path": str(whisper_dir),
            "ready": whisper_ready,
        },
    }


class UpdateEnvSettingsRequest(BaseModel):
    values: Dict[str, str]


class DownloadModelRequest(BaseModel):
    target: str = "all"  # embedding | reranker | whisper | all
    hf_endpoint: Optional[str] = None


@app.get("/api/settings/env")
async def get_env_settings(request: Request):
    _require_localhost(request)
    env_map = _read_env_file_map(ENV_PATH)
    values: Dict[str, Any] = {}
    secrets_configured: Dict[str, str] = {}
    for key in sorted(_SETTINGS_ALLOWED_KEYS):
        val = env_map.get(key)
        if val is None:
            val = os.getenv(key, "")
        if key in _SETTINGS_SECRET_KEYS:
            values[key] = ""
            secrets_configured[key] = _mask_secret(val)
        else:
            values[key] = val
    return {
        "env_path": str(ENV_PATH),
        "values": values,
        "secrets_configured": secrets_configured,
        "models_status": _check_local_models_status(),
    }


@app.post("/api/settings/env")
async def update_env_settings(req: UpdateEnvSettingsRequest, request: Request):
    _require_localhost(request)
    current_map = _read_env_file_map(ENV_PATH)
    to_write: Dict[str, str] = {}
    restart_required = False

    for k, raw_v in (req.values or {}).items():
        if k not in _SETTINGS_ALLOWED_KEYS:
            continue
        v = str(raw_v or "").strip()
        if k in _SETTINGS_SECRET_KEYS and not v:
            # 留空表示保留原密钥，不覆盖清空
            continue
        old_val = current_map.get(k, os.getenv(k, ""))
        to_write[k] = v
        os.environ[k] = v
        if k in _SETTINGS_RESTART_KEYS and v != old_val:
            restart_required = True

    if to_write:
        _write_env_file_updates(ENV_PATH, to_write)

    return {
        "status": "saved",
        "updated_keys": list(to_write.keys()),
        "restart_required": restart_required,
        "model_name": get_default_llm_model(),
        "models_status": _check_local_models_status(),
    }


@app.post("/api/settings/models/download")
async def start_model_download(req: DownloadModelRequest, request: Request):
    _require_localhost(request)
    target = (req.target or "all").strip().lower()
    if target not in {"embedding", "reranker", "whisper", "all"}:
        raise HTTPException(status_code=400, detail=f"不支持的模型目标: {target}")

    with _MODEL_DOWNLOAD_LOCK:
        if _MODEL_DOWNLOAD_STATE["running"]:
            raise HTTPException(status_code=409, detail=f"已有下载任务正在运行 ({_MODEL_DOWNLOAD_STATE['target']})，请等待完成")
        _MODEL_DOWNLOAD_STATE.update({
            "running": True,
            "target": target,
            "status": "running",
            "returncode": None,
            "logs": [f"🚀 启动模型下载任务: {target}"],
        })

    hf_endpoint = (req.hf_endpoint or os.getenv("HF_ENDPOINT") or "").strip()

    def _run_download():
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if hf_endpoint:
            env["HF_ENDPOINT"] = hf_endpoint
        cmd = [sys.executable, str(BASE_DIR / "download_vector_models.py"), target]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                clean = line.rstrip()
                if not clean:
                    continue
                with _MODEL_DOWNLOAD_LOCK:
                    _MODEL_DOWNLOAD_STATE["logs"].append(clean)
                    if len(_MODEL_DOWNLOAD_STATE["logs"]) > 200:
                        _MODEL_DOWNLOAD_STATE["logs"] = _MODEL_DOWNLOAD_STATE["logs"][-200:]
            rc = proc.wait()
            with _MODEL_DOWNLOAD_LOCK:
                _MODEL_DOWNLOAD_STATE["running"] = False
                _MODEL_DOWNLOAD_STATE["returncode"] = rc
                _MODEL_DOWNLOAD_STATE["status"] = "completed" if rc == 0 else "failed"
                _MODEL_DOWNLOAD_STATE["logs"].append(
                    "✅ 模型下载完成！（若为首次下载，建议重启后端服务加载新权重）" if rc == 0 else f"❌ 模型下载退出码: {rc}"
                )
        except Exception as exc:
            with _MODEL_DOWNLOAD_LOCK:
                _MODEL_DOWNLOAD_STATE["running"] = False
                _MODEL_DOWNLOAD_STATE["returncode"] = -1
                _MODEL_DOWNLOAD_STATE["status"] = "failed"
                _MODEL_DOWNLOAD_STATE["logs"].append(f"❌ 下载异常: {exc}")

    threading.Thread(target=_run_download, daemon=True).start()
    return {"status": "started", "target": target}


@app.get("/api/settings/models/download/status")
async def get_model_download_status(request: Request):
    _require_localhost(request)
    with _MODEL_DOWNLOAD_LOCK:
        snapshot = {
            "running": _MODEL_DOWNLOAD_STATE["running"],
            "target": _MODEL_DOWNLOAD_STATE["target"],
            "status": _MODEL_DOWNLOAD_STATE["status"],
            "returncode": _MODEL_DOWNLOAD_STATE["returncode"],
            "logs": list(_MODEL_DOWNLOAD_STATE["logs"][-80:]),
        }
    snapshot["models_status"] = _check_local_models_status()
    return snapshot


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": SessionManager.list_sessions()}


@app.post("/api/sessions/create")
async def create_session(req: CreateSessionRequest):
    session = SessionManager.create_session(title=req.title, domain=req.domain)
    return {"session": session.model_dump()}


@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str):
    session = SessionManager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session.model_dump()}


class RenameSessionRequest(BaseModel):
    title: str


@app.put("/api/sessions/{session_id}")
async def rename_session(session_id: str, req: RenameSessionRequest):
    session = SessionManager.rename_session(session_id, req.title)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session.model_dump()}


@app.post("/api/sessions/{session_id}/clear")
async def clear_session_messages(session_id: str):
    with SessionManager.session_lock(session_id):
        session = SessionManager.clear_session_messages(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        GLOBAL_MEMORY_SAVER.delete_thread(session_id)
    return {"session": session.model_dump()}


@app.post("/api/sessions/clear_all")
async def clear_all_sessions():
    for item in SessionManager.list_sessions():
        GLOBAL_MEMORY_SAVER.delete_thread(item["session_id"])
    init_s = SessionManager.clear_all_sessions()
    return {"initial_session": init_s.model_dump()}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    with SessionManager.session_lock(session_id):
        ok = SessionManager.delete_session(session_id)
        if ok:
            GLOBAL_MEMORY_SAVER.delete_thread(session_id)
    return {"success": ok}


# ==========================================
# 2. 聊天与问答接口 (支持多会话隔离)
# ==========================================
class ChatRequest(BaseModel):
    query: str
    image_data: Optional[str] = None  # Base64 附带截图或图片 (可为空)
    attached_doc: Optional[str] = None  # 输入框随提问附带的文档文件名 (PDF/Word/PPT/TXT/MD 等)
    session_id: Optional[str] = None
    context_hint: Optional[str] = None
    voice: Optional[str] = None
    meeting_record_path: Optional[str] = None


def _execute_chat_workflow(req: ChatRequest) -> Dict[str, Any]:
    session_id = req.session_id
    if session_id and session_id != "null":
        with SessionManager.session_lock(session_id):
            return _execute_chat_workflow_locked(req)
    return _execute_chat_workflow_locked(req)


def _execute_chat_workflow_locked(req: ChatRequest) -> Dict[str, Any]:
    session_id = req.session_id
    if not session_id or session_id == "null":
        new_sess = SessionManager.create_session(title=req.query.strip()[:24])
        session_id = new_sess.session_id

    # 获取该会话的长期研讨摘要、聚焦研讨文献与主题生命周期状态机
    session = SessionManager.get_session(session_id)
    history_summary = ""
    active_doc = SessionManager.get_active_doc(session_id)
    topic_state = SessionManager.get_topic_state(session_id)
    
    # 外科手术式在历史中锁定最合适的主题定论 (0ms 极速寻址，拒绝全量地毯式盲搜)
    matched_topic = SessionManager.match_best_historical_topic(session_id, query=req.query)
    history_summary = SessionManager.get_scoped_thematic_context(session_id, query=req.query)

    # 保存用户发送的消息到指定领域频道 (包含附带截图/文档)
    user_display_query = f"{req.query}\n[附带解析文档: {req.attached_doc}]" if req.attached_doc and req.attached_doc not in req.query else req.query
    user_msg = ChatMessage(sender="user", content=user_display_query, image_url=req.image_data)
    SessionManager.add_message(session_id, user_msg)

    attached_evidence: List[Dict[str, Any]] = []
    if req.attached_doc:
        active_doc = req.attached_doc
        SessionManager.set_active_doc(session_id, active_doc)
        MAX_ATTACHED_DOC_CHARS = 120000
        doc_path = UPLOAD_DIR / req.attached_doc
        parsed_chunks = []
        if doc_path.exists() and doc_path.is_file():
            try:
                from document_rag_engine import UniversalDocumentParser
                parsed_chunks = UniversalDocumentParser.parse_file(doc_path)
            except Exception as parse_exc:
                print(f"⚠️ [AttachedDoc Parse Note]: {parse_exc}")

        total_parsed_chars = sum(len((c.text or "").strip()) for c in parsed_chunks)
        if parsed_chunks and total_parsed_chars <= MAX_ATTACHED_DOC_CHARS:
            # 常规文档（<= 12万字）：全量送入上下文，确保 AI 完整阅读全文无遗漏
            for chunk in parsed_chunks:
                txt = (chunk.text or "").strip()
                if not txt:
                    continue
                attached_evidence.append({
                    "doc_name": chunk.doc_name,
                    "doc_type": chunk.doc_type,
                    "location_label": f"{chunk.page_or_slide} · {chunk.section}".strip(" ·"),
                    "text": txt,
                    "summary": txt[:300],
                    "score": 0.99,
                    "evidence_level": "local_full_doc",
                    "is_local_indexed": True,
                })
        else:
            # 超长巨著（> 12万字）：先按用户提问做全库语义精排召回，再补足开篇与结构片段直至预算上限
            seen_texts = set()
            acc_chars = 0
            try:
                rag_hits = rag_engine.search(
                    query=req.query,
                    top_k=15,
                    candidate_k=40,
                    doc_filter=req.attached_doc,
                )
                for hit in rag_hits or []:
                    txt = (hit.get("text") or "").strip()
                    key = txt[:120]
                    if txt and key not in seen_texts and acc_chars + len(txt) <= MAX_ATTACHED_DOC_CHARS:
                        seen_texts.add(key)
                        acc_chars += len(txt)
                        attached_evidence.append({
                            "doc_name": hit.get("doc_name") or req.attached_doc,
                            "doc_type": hit.get("doc_type") or Path(req.attached_doc).suffix.lstrip("."),
                            "location_label": hit.get("location_label") or "",
                            "text": txt,
                            "summary": hit.get("summary") or txt[:300],
                            "score": max(0.92, float(hit.get("score") or 0.92)),
                            "evidence_level": "local_snippet",
                            "is_local_indexed": True,
                        })
            except Exception as search_exc:
                print(f"⚠️ [AttachedDoc Search Note]: {search_exc}")

            for chunk in parsed_chunks:
                txt = (chunk.text or "").strip()
                key = txt[:120]
                if not txt or key in seen_texts:
                    continue
                if acc_chars + len(txt) > MAX_ATTACHED_DOC_CHARS:
                    break
                seen_texts.add(key)
                acc_chars += len(txt)
                attached_evidence.append({
                    "doc_name": chunk.doc_name,
                    "doc_type": chunk.doc_type,
                    "location_label": f"{chunk.page_or_slide} · {chunk.section}".strip(" ·"),
                    "text": txt,
                    "summary": txt[:300],
                    "score": 0.90,
                    "evidence_level": "local_snippet",
                    "is_local_indexed": True,
                })

    recent_candidates = SessionManager.get_recent_candidates(session_id)
    initial_state: MasterAgentState = {
        "session_id": session_id,
        "user_query": req.query,
        "task_snapshot": None,
        "extracted_visual_context": None,
        "delivery_checks": None,
        "control_result": None,
        "specialist_results": [],
        "meeting_record_path": req.meeting_record_path or next((m.meeting_record_path
            for m in reversed(session.messages if session else []) if m.meeting_record_path), None),
        "retrieval_plan": None,
        "retrieval_queries": [],
        "retrieval_probes": [],
        "retrieval_coverage": None,
        "source_diagnostics": None,
        "review_checks": [],
        "review_notice": None,
        "user_image": req.image_data,
        "intent_decision": None,
        "retrieved_evidence": attached_evidence,
        "final_answer": "",
        "target_agent": "",
        "tts_audio_file": None,
        "tts_voice": req.voice or (session.tts_voice if session else "zh-CN-XiaoyiNeural"),
        "history_summary": history_summary,
        "active_doc": active_doc,
        "topic_state": topic_state,
        "recent_paper_candidates": recent_candidates,
        "shared_resources": [],
        "shared_pool": SessionManager.get_shared_pool(session_id, include_content=False),
        "topic_catalog": [],
        "matched_topic": matched_topic,
        "pending_action": None,
        "synthesized_evidence_slate": None,
        "dialectical_notes": None,
        "reflection_notes": None,
        "peer_review_notes": None,
        "review_verdict": None,
        "review_scores_round1": None,
        "revision_count": 0,
        "pending_evidence_additions": [],
        "evidence_append_count": 0,
        "review_retrieval_rounds": 0,
        "review_retrieval_status": None,
        "review_retrieval_log": [],
        "agent_dialogue_history": [],
        "status": "init",
    }

    # 执行 LangGraph 状态图 (原生挂载 thread_id 记忆会话检查点)
    try:
        # Retrieval may follow several task-specific gaps before reaching the
        # safety budget. Keep LangGraph's structural limit above that budget.
        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 100}
        with system_control_executor(lambda action, params: execute_system_control(
                action, params, session_id=session_id, voice_manager=voice_manager, session_manager=SessionManager)):
            result_state = master_app.invoke(initial_state, config=config)
    except Exception as e:
        print(f"❌ [Master Workflow Error]: {e}")
        err_msg = str(e)
        if "503" in err_msg or "Service temporarily unavailable" in err_msg:
            fallback_text = "⚠️ **【搭档提示】上游大模型 API 出现暂时性繁忙排队 (HTTP 503 Service Unavailable)**\n\n上游服务暂时过载，请稍候点击下方快捷词条或重新发送问题即可继续研讨。"
        elif "429" in err_msg or "Rate limit" in err_msg:
            fallback_text = "⚠️ **【搭档提示】模型调用频次已达限制 (HTTP 429 Rate Limit)**\n\n请稍候 3~5 秒后再次提问。"
        elif "524" in err_msg:
            fallback_text = "⚠️ **【搭档提示】上游代理响应超时 (HTTP 524 Gateway Timeout)**\n\n上游代理排队较慢，请稍后重试。"
        else:
            fallback_text = f"⚠️ **【搭档提示】大模型接口调用异常**: {err_msg}\n\n建议稍后重新发送提问。"

        result_state = {
            "intent_decision": None,
            "target_agent": "ErrorFallbackAgent",
            "topic_state": topic_state,
            "retrieved_evidence": [],
            "final_answer": fallback_text,
            "tts_audio_file": None,
            "status": "error_fallback",
        }

    control = result_state.get("control_result") or {}
    if control.get("action") == "clear_session" and control.get("status") == "pending":
        # Clear only after the graph stops writing checkpoints; do not publish
        # the old task or answer into the freshly emptied session.
        try:
            cleared = SessionManager.clear_session_messages(session_id)
            if cleared is not None:
                GLOBAL_MEMORY_SAVER.delete_thread(session_id)
            status = "completed" if cleared is not None else "not_executed"
            message = "已清空当前会话。" if cleared is not None else "未找到当前会话，未执行清空。"
        except Exception:
            status, message = "failed", "当前会话清空未完整完成，请重试。"
        return {"session_id": session_id, "query": req.query, "final_answer": message,
                "target_agent": "SystemControlAgent", "retrieved_evidence": [], "audio_url": None,
                "control_result": {"action": "clear_session", "status": status, "message": message},
                "status": "system_control_" + status}

    # 持久化同步主题生命周期状态机与前序学术候选池
    if result_state.get("topic_state"):
        SessionManager.update_topic_state(session_id, result_state["topic_state"])

    if result_state.get("recent_paper_candidates"):
        SessionManager.update_recent_candidates(session_id, result_state["recent_paper_candidates"])
    if result_state.get("retrieved_evidence"):
        SessionManager.update_shared_resources(session_id, result_state["retrieved_evidence"])
    selected_shared_ids = (result_state.get("retrieval_plan") or {}).get("selected_shared_item_ids") or []
    SessionManager.touch_shared_pool_items(session_id, selected_shared_ids)

    if result_state.get("active_doc"):
        s_obj = SessionManager.get_session(session_id)
        if s_obj:
            s_obj.active_doc = result_state["active_doc"]
            SessionManager.save_session(s_obj)

    audio_url = None
    if result_state.get("tts_audio_file") and os.path.exists(result_state["tts_audio_file"]):
        audio_name = Path(result_state["tts_audio_file"]).name
        audio_url = f"/api/audio/{audio_name}"

    from document_rag_engine import extract_answer_mindmap
    from retrieval_planner import finalize_answer_markdown, renumber_citations_and_evidence
    _raw_evidence = result_state.get("retrieved_evidence") or []
    _raw_candidates = result_state.get("recent_paper_candidates") or []
    _renumbered_ans, _ordered_evidence = renumber_citations_and_evidence(
        result_state.get("final_answer", ""),
        _raw_evidence,
    )
    _sync_snapshot = [
        dict(item) for item in (_ordered_evidence if any(item.get("text") or item.get("summary") for item in _ordered_evidence if isinstance(item, dict)) else _raw_candidates)
        if isinstance(item, dict)
    ]
    final_ans = finalize_answer_markdown(
        _renumbered_ans,
        _ordered_evidence,
        _raw_candidates,
    )
    mindmap_md = extract_answer_mindmap(final_ans, req.query)
    turn_snapshot = {
        key: result_state.get(key)
        for key in (
            "topic_state", "active_doc", "active_subject", "sub_subject",
            "subject_evolution_log", "status",
        )
        if result_state.get(key) is not None
    }

    # 保存智能体回复到该领域频道
    agent_msg = ChatMessage(
        sender="agent",
        content=final_ans,
        intent_decision=result_state.get("intent_decision"),
        task_snapshot=result_state.get("task_snapshot"),
        delivery_checks=result_state.get("delivery_checks"),
        specialist_results=result_state.get("specialist_results") or [],
        meeting_record_path=result_state.get("meeting_record_path"),
        control_result=result_state.get("control_result"),
        target_agent=result_state.get("target_agent"),
        retrieved_evidence=_ordered_evidence,
        mindmap_markdown=mindmap_md,
        audio_url=audio_url,
        agent_dialogue_history=result_state.get("agent_dialogue_history", []),
        synthesized_evidence_slate=result_state.get("synthesized_evidence_slate"),
        dialectical_notes=result_state.get("dialectical_notes") or result_state.get("peer_review_notes"),
        review_verdict=result_state.get("review_verdict"),
        review_checks=result_state.get("review_checks", []),
        review_notice=result_state.get("review_notice"),
        review_retrieval_log=result_state.get("review_retrieval_log", []),
        retrieval_plan=result_state.get("retrieval_plan"),
        retrieval_coverage=result_state.get("retrieval_coverage"),
        source_diagnostics=result_state.get("source_diagnostics"),
    )
    SessionManager.add_message(session_id, agent_msg)
    
    # Publish the turn once into the shared pool; the pool is also the historical topic index.
    s_now = SessionManager.get_session(session_id)
    turn_num = len([m for m in (s_now.messages if s_now else []) if m.sender == "agent"])
    SessionManager.publish_turn_to_shared_pool(session_id, turn_num, req.query, agent_msg, turn_state=turn_snapshot)

    if os.getenv("ENABLE_AUTO_SYNC_ONLINE_EVIDENCE", "1") != "0" and _sync_snapshot:
        import re as _re
        import threading as _threading
        _cited_indices = [int(m.group(1)) for m in _re.finditer(r"\[(\d+)\]", final_ans)]

        def _bg_sync_online_evidence():
            try:
                from master_agent_workflow import MasterEngineRegistry
                MasterEngineRegistry.get_rag_engine().ingest_online_evidence(
                    _sync_snapshot,
                    cited_indices=_cited_indices,
                )
            except Exception as _sync_exc:
                print(f"⚠️ [AutoSync Note]: {_sync_exc}")

        _threading.Thread(target=_bg_sync_online_evidence, daemon=True).start()

    return {
        "session_id": session_id,
        "query": req.query,
        "history_summary": history_summary,
        "topic_state": result_state.get("topic_state"),
        "intent_decision": result_state.get("intent_decision"),
        "task_snapshot": result_state.get("task_snapshot"),
        "delivery_checks": result_state.get("delivery_checks"),
        "specialist_results": result_state.get("specialist_results") or [],
        "meeting_record_path": result_state.get("meeting_record_path"),
        "control_result": result_state.get("control_result"),
        "target_agent": result_state.get("target_agent"),
        "retrieved_evidence": _ordered_evidence,
        "retrieval_plan": result_state.get("retrieval_plan"),
        "retrieval_coverage": result_state.get("retrieval_coverage"),
        "source_diagnostics": result_state.get("source_diagnostics"),
        "final_answer": final_ans,
        "mindmap_markdown": mindmap_md,
        "audio_url": audio_url,
        "active_doc": result_state.get("active_doc") or active_doc,
        "matched_topic": matched_topic,
        "pending_action": result_state.get("pending_action"),
        "synthesized_evidence_slate": result_state.get("synthesized_evidence_slate"),
        "dialectical_notes": result_state.get("dialectical_notes") or result_state.get("peer_review_notes"),
        "reflection_notes": result_state.get("reflection_notes"),
        "peer_review_notes": result_state.get("peer_review_notes"),
        "review_verdict": result_state.get("review_verdict"),
        "review_checks": result_state.get("review_checks", []),
        "review_notice": result_state.get("review_notice"),
        "review_retrieval_log": result_state.get("review_retrieval_log", []),
        "review_scores_round1": result_state.get("review_scores_round1"),
        "agent_dialogue_history": result_state.get("agent_dialogue_history", []),
        "status": result_state.get("status"),
    }


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return _execute_chat_workflow(req)


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def token_handler(token: str):
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "token", "content": token})

    async def run_workflow():
        try:
            register_stream_callback(token_handler)
            result = await asyncio.to_thread(_execute_chat_workflow, req)
            register_stream_callback(None)
            await queue.put({"type": "done", "data": result})
        except Exception as e:
            register_stream_callback(None)
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put(None)

    asyncio.create_task(run_workflow())

    async def sse_generator():
        yield f"data: {json.dumps({'type': 'start', 'query': req.query}, ensure_ascii=False)}\n\n"
        while True:
            msg = await queue.get()
            if msg is None:
                break
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )



# ==========================================
# 2. 文档上传与知识库构建 (PDF / Word / PPT)
# ==========================================
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    
    dest_path = UPLOAD_DIR / file.filename
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 自动解析切块入库 (依托全格式解析与最大宽容度兜底)
    try:
        chunks_count = rag_engine.ingest_file(dest_path)
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=400, detail=f"文件解析入库失败 ({ext}): {e}")

    return {
        "filename": file.filename,
        "doc_type": ext.lstrip(".") or "txt",
        "chunks_count": chunks_count,
        "status": "ingested",
    }


@app.get("/api/documents")
async def list_documents():
    docs = []
    # 过滤掉元数据 json、临时文件以及隐藏文件，仅呈现正规文献
    ignored_suffixes = {".json", ".tmp", ".bak", ".part", ".crdownload"}
    for f in UPLOAD_DIR.glob("*.*"):
        if f.is_file() and f.suffix.lower() not in ignored_suffixes and not f.name.startswith("."):
            docs.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "modified_at": f.stat().st_mtime,
                "type": f.suffix.lstrip(".").lower()
            })
    docs.sort(key=lambda x: x["modified_at"], reverse=True)
    synced_papers = rag_engine.list_online_synced_papers() if hasattr(rag_engine, "list_online_synced_papers") else []
    return {"documents": docs, "synced_papers": synced_papers}


@app.get("/api/documents/synced")
async def list_synced_documents():
    """列出本地知识库中自动沉淀的在线高相关论文 (doc_type=online_synced_paper)"""
    papers = rag_engine.list_online_synced_papers() if hasattr(rag_engine, "list_online_synced_papers") else []
    return {"synced_papers": papers, "count": len(papers)}


# ==========================================
# 2.1 经卷精读：多模态原文阅读与划词/框选伴读接口
# ==========================================
def _resolve_upload_doc(filename: str) -> Path:
    safe_name = Path(filename or "").name
    if not safe_name:
        raise HTTPException(status_code=400, detail="缺少文件名")
    doc_path = (UPLOAD_DIR / safe_name).resolve()
    if not str(doc_path).startswith(str(UPLOAD_DIR.resolve())) or not doc_path.exists() or not doc_path.is_file():
        raise HTTPException(status_code=404, detail=f"未找到文档: {safe_name}")
    return doc_path


@app.get("/api/reader/meta")
async def reader_doc_meta(filename: str):
    """获取文档阅读元数据：PDF 返回页数与目录大纲，Word/PPT/EPUB/MD 返回章节结构"""
    doc_path = _resolve_upload_doc(filename)
    ext = doc_path.suffix.lower()
    if ext == ".pdf":
        import fitz
        with fitz.open(str(doc_path)) as doc:
            toc = [
                {"level": int(item[0]), "title": str(item[1]).strip(), "page": int(item[2])}
                for item in (doc.get_toc(simple=True) or [])
                if len(item) >= 3 and str(item[1]).strip()
            ][:120]
            page_aspect = 1.414
            if len(doc) > 0:
                r0 = doc[0].rect
                if r0.width > 0 and r0.height > 0:
                    page_aspect = round(float(r0.height / r0.width), 4)
            return {
                "filename": doc_path.name,
                "mode": "pdf",
                "page_count": len(doc),
                "page_aspect": page_aspect,
                "toc": toc,
                "size_bytes": doc_path.stat().st_size,
            }
    from document_rag_engine import UniversalDocumentParser
    chunks = UniversalDocumentParser.parse_file(doc_path)
    sections = [
        {
            "index": i + 1,
            "location_label": f"{c.page_or_slide} · {c.section}".strip(" ·") or f"第 {i + 1} 节",
            "text": (c.text or "").strip(),
        }
        for i, c in enumerate(chunks)
        if (c.text or "").strip()
    ]
    return {
        "filename": doc_path.name,
        "mode": "sections",
        "section_count": len(sections),
        "sections": sections,
        "size_bytes": doc_path.stat().st_size,
    }


@app.get("/api/reader/pdf_page")
async def reader_pdf_page(filename: str, page: int = 1, scale: float = 2.2):
    """按高 DPI 原貌渲染 PDF 单页为 PNG，完整保留扫描件、图表与复杂公式"""
    doc_path = _resolve_upload_doc(filename)
    if doc_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="仅 PDF 支持逐页图像渲染")
    import fitz
    clamped_scale = max(1.0, min(float(scale or 2.2), 3.5))
    with fitz.open(str(doc_path)) as doc:
        if page < 1 or page > len(doc):
            raise HTTPException(status_code=400, detail=f"页码越界 (1~{len(doc)})")
        pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(clamped_scale, clamped_scale), alpha=False)
        img_bytes = pix.tobytes("png")
    return Response(
        content=img_bytes,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/api/reader/pdf_page_text")
async def reader_pdf_page_text(filename: str, page: int = 1):
    """提取 PDF 指定页的文本层与逐行归一化坐标（供前端透明文本层直接划选复制）"""
    doc_path = _resolve_upload_doc(filename)
    if doc_path.suffix.lower() != ".pdf":
        return {"page": page, "text": "", "lines": []}
    import fitz
    lines_out = []
    with fitz.open(str(doc_path)) as doc:
        if page < 1 or page > len(doc):
            raise HTTPException(status_code=400, detail=f"页码越界 (1~{len(doc)})")
        page_obj = doc[page - 1]
        txt = (page_obj.get_text("text") or "").strip()
        pw = float(page_obj.rect.width or 1.0)
        ph = float(page_obj.rect.height or 1.0)
        text_dict = page_obj.get_text("dict") or {}
        for blk in text_dict.get("blocks", []):
            if blk.get("type") != 0:
                continue
            for ln in blk.get("lines", []):
                # 过滤竖排/旋转边栏水印（如 arXiv 左侧 90° 竖排编号），避免超高包围盒遮挡正文划选
                ldir = ln.get("dir")
                if ldir and len(ldir) == 2:
                    dx, dy = abs(float(ldir[0])), abs(float(ldir[1]))
                    if dx < 0.7 or dy > 0.5:
                        continue
                spans = ln.get("spans", [])
                line_text = "".join(str(sp.get("text", "")) for sp in spans)
                if not line_text.strip():
                    continue
                bbox = ln.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = [float(v) for v in bbox]
                w = max(0.1, (x1 - x0) / pw * 100.0)
                h = max(0.1, (y1 - y0) / ph * 100.0)
                if h > 12.0:
                    continue
                lines_out.append({
                    "t": line_text,
                    "x": round(max(0.0, x0 / pw * 100.0), 4),
                    "y": round(max(0.0, y0 / ph * 100.0), 4),
                    "w": round(w, 4),
                    "h": round(h, 4),
                })
    return {"page": page, "text": txt, "lines": lines_out}


class ReaderAskRequest(BaseModel):
    filename: str
    page: Optional[int] = None
    selected_text: Optional[str] = None
    crop_image_data: Optional[str] = None
    question: str = ""
    model_mode: str = "fast"  # fast | primary
    history: List[Dict[str, str]] = []


def _run_reader_multimodal_ask(req: ReaderAskRequest) -> Dict[str, Any]:
    from openai import OpenAI
    from config_utils import (
        get_default_llm_model,
        get_llm_timeout,
        get_translation_llm_client_kwargs,
        get_translation_llm_model,
    )

    doc_path = _resolve_upload_doc(req.filename)
    page_text = ""
    if doc_path.suffix.lower() == ".pdf" and req.page:
        try:
            import fitz
            with fitz.open(str(doc_path)) as doc:
                if 1 <= req.page <= len(doc):
                    page_text = (doc[req.page - 1].get_text("text") or "").strip()[:3500]
        except Exception:
            page_text = ""

    # 局部同书检索：补充上下文符号/定义（毫秒级，不触发外网检索）
    local_snippets: List[str] = []
    search_seed = (req.question or req.selected_text or "").strip()
    if search_seed:
        try:
            hits = rag_engine.search(query=search_seed[:400], top_k=3, candidate_k=10, doc_filter=doc_path.name)
            for h in hits or []:
                t = (h.get("text") or "").strip()
                if t:
                    loc = h.get("location_label") or ""
                    local_snippets.append(f"[{loc}] {t[:600]}")
        except Exception:
            pass

    use_fast = (req.model_mode or "fast").lower() == "fast" and bool(os.getenv("TRANSLATION_LLM_MODEL", "").strip())
    if use_fast:
        client_kwargs = get_translation_llm_client_kwargs()
        model_name = get_translation_llm_model()
    else:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        model_name = get_default_llm_model()

    if not client_kwargs.get("api_key"):
        raise ValueError("未配置 OPENAI_API_KEY，请在右上角 ⚙️ 系统设置中填写")

    client = OpenAI(**client_kwargs)
    system_prompt = (
        "你是严谨、直觉敏锐的学术原文伴读助手。用户正在精读一篇论文或专著，可能会划选段落，或直接框选扫描件/模糊公式/图表截图提问。\n"
        "要求：\n"
        "1. 若附带了局部截图（含扫描件、模糊文字、数学公式或图表），请优先仔细辨识图片中的真实符号、坐标轴、数值与公式细节，绝不凭空臆造；若图片局部确实过于模糊无法确认某个符号，请如实指出。\n"
        "2. 讲解兼顾直觉机制与严谨推导：先用清晰中文讲清核心物理/逻辑含义，再用标准 LaTeX（行内 $...$，独立公式 $$...$$）展开必要推导或解释变量。\n"
        "3. 回答紧扣当前选中内容与同书上下文，精炼清晰、不堆砌空话。"
    )

    context_parts = [f"当前精读文档：《{doc_path.name}》" + (f"（第 {req.page} 页）" if req.page else "")]
    if req.selected_text:
        context_parts.append(f"【用户选中原文】：\n{req.selected_text.strip()[:4000]}")
    if page_text:
        context_parts.append(f"【本页文本层参考】：\n{page_text}")
    if local_snippets:
        context_parts.append("【同文档相关背景切片】：\n" + "\n\n".join(local_snippets))
    user_q = (req.question or "").strip() or (
        "请仔细识别图中内容（若是扫描件/公式/图表，请先还原关键文字与公式），并清晰解释其含义、推导逻辑与核心结论。"
        if req.crop_image_data
        else "请清晰解读这段原文的核心含义、关键术语/公式及其在上下文中的作用。"
    )
    context_parts.append(f"【用户提问】：{user_q}")
    full_user_text = "\n\n".join(context_parts)

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for h in (req.history or [])[-6:]:
        role = h.get("role")
        content = (h.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:2500]})

    if req.crop_image_data and str(req.crop_image_data).startswith("data:image/"):
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": full_user_text},
                {"type": "image_url", "image_url": {"url": req.crop_image_data}},
            ],
        })
    else:
        messages.append({"role": "user", "content": full_user_text})

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.25,
        timeout=get_llm_timeout(120.0),
    )
    answer = (resp.choices[0].message.content or "").strip()
    return {
        "answer": answer,
        "model": model_name,
        "page": req.page,
        "has_image": bool(req.crop_image_data),
    }


@app.post("/api/reader/ask")
async def reader_multimodal_ask(req: ReaderAskRequest):
    """经卷精读多模态伴读问答（默认轻快单次多模态调用，不打断阅读心流）"""
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _run_reader_multimodal_ask, req)
        return {"status": "success", "data": data}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"伴读解答失败: {exc}")


@app.delete("/api/documents/synced")
async def clear_synced_documents(doc_name: Optional[str] = None):
    """删除单篇或一键清空所有自动沉淀的在线论文记忆（不影响用户上传的物理文档）"""
    deleted = rag_engine.clear_online_synced_papers(doc_name=doc_name) if hasattr(rag_engine, "clear_online_synced_papers") else 0
    return {"deleted_count": deleted, "doc_name": doc_name, "success": True}


@app.get("/api/documents/{doc_name}/mindmap")
async def get_doc_mindmap(doc_name: str):
    """获取指定文献的深层细分全书/全篇经卷思维导图"""
    mindmap_md = rag_engine.get_document_mindmap(doc_name)
    return {"doc_name": doc_name, "mindmap_markdown": mindmap_md}


@app.get("/api/documents/{doc_name}/download")
async def download_document(doc_name: str):
    """下载或直接阅读本地琅嬛书阁中已入库的文献原卷 (PDF / Office / Text)"""
    target_file = UPLOAD_DIR / doc_name
    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail="文献物理文件不存在")
    
    suffix = target_file.suffix.lower()
    media_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".txt": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
    }
    media_type = media_map.get(suffix, "application/octet-stream")
    return FileResponse(
        path=target_file,
        filename=target_file.name,
        media_type=media_type,
    )


@app.delete("/api/documents/{doc_name}")
async def delete_document(doc_name: str):
    """彻底删除指定文献，同时清除 ChromaDB 向量库中对应切片与磁盘物理文件"""
    # 1. 从 ChromaDB 向量库彻底删除切片
    deleted_from_db = rag_engine.delete_document(doc_name)

    # 2. 从 uploads/ 目录删除物理文件
    target_file = UPLOAD_DIR / doc_name
    deleted_from_disk = False
    if target_file.exists():
        target_file.unlink(missing_ok=True)
        deleted_from_disk = True

    return {
        "doc_name": doc_name,
        "deleted_from_db": deleted_from_db,
        "deleted_from_disk": deleted_from_disk,
        "success": True,
    }


# ==========================================
# 3. 会议记录与纪要 (Meeting & Action Items)
# ==========================================
@app.get("/api/meetings/latest")
async def get_latest_meeting():
    jsonl_files = list(RECORDS_DIR.glob("record_*.jsonl"))
    if not jsonl_files:
        return {"has_meeting": False, "report": None, "raw_records": []}

    jsonl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_file = jsonl_files[0]

    # 读取最近 30 条发言流水
    raw_lines = []
    with open(latest_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    raw_lines.append(json.loads(line.strip()))
                except Exception:
                    pass

    return {
        "has_meeting": True,
        "filename": latest_file.name,
        "total_speeches": len(raw_lines),
        "recent_speeches": raw_lines[-20:],
    }


@app.post("/api/meetings/summarize")
async def trigger_meeting_summary():
    jsonl_files = list(RECORDS_DIR.glob("record_*.jsonl"))
    if not jsonl_files:
        raise HTTPException(status_code=404, detail="No meeting records found")

    jsonl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    report = meeting_summarizer.summarize_session(jsonl_files[0])
    return {"report": report.model_dump()}


# ==========================================
# 3.1 Obsidian 知识库与 Canvas 白板一键导出
# ==========================================
class ObsidianExportRequest(BaseModel):
    session_id: str
    export_canvas: bool = True
    custom_vault_path: Optional[str] = None


@app.post("/api/export/obsidian")
async def export_to_obsidian(req: ObsidianExportRequest):
    """将当前会话全量导出为 Obsidian 双链笔记 (.md) 与原生白板 (.canvas)"""
    from obsidian_exporter import ObsidianKnowledgeExporter
    vault_p = Path(req.custom_vault_path) if req.custom_vault_path else None
    exporter = ObsidianKnowledgeExporter(vault_path=vault_p)
    result = exporter.export_session(session_id=req.session_id, export_canvas=req.export_canvas)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


# ==========================================
# 4. 实时语音感知与时间轴流式推送 (Live Voice Stream & WebSocket)
# ==========================================
class VoiceSessionManager:
    def __init__(self):
        self.is_running = False
        self.agent_thread = None
        self.connected_websockets: List[WebSocket] = []
        self.event_loop = None
        import threading
        self._control_lock = threading.RLock()

    async def broadcast_speech(self, entry: Dict[str, Any]):
        dead_sockets = []
        for ws in self.connected_websockets:
            try:
                await ws.send_json(entry)
            except Exception:
                dead_sockets.append(ws)
        for ws in dead_sockets:
            if ws in self.connected_websockets:
                self.connected_websockets.remove(ws)

    def start_voice_agent(self, loop, mode: str = "mic"):
        with self._control_lock:
            return self._start_voice_agent(loop, mode)

    def _start_voice_agent(self, loop, mode: str):
        if self.is_running:
            return {"status": "completed", "message": "录音监听已在运行。", "is_recording": True}
        if self.agent_thread is not None and self.agent_thread.is_alive():
            return {"status": "stopping", "message": "上一轮录音仍在结束，请稍后再开启。", "is_recording": False}
        self.is_running = True
        self.event_loop = loop
        mode = mode if mode in {"mic", "system"} else "mic"

        def run_worker():
            try:
                from datetime import datetime
                from voice_langgraph_agent import VoicePerceptionAgent
                agent = VoicePerceptionAgent(source="default", mode=mode)
                
                # 回调推送到 WebSocket
                orig_clean_and_format = agent._clean_and_format
                def hooked_clean_and_format(state):
                    res = orig_clean_and_format(state)
                    entry = {
                        "type": "speech",
                        "index": state["current_index"],
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "text": state["cleaned_text"],
                        "source": state.get("active_source", "mic")
                    }
                    asyncio.run_coroutine_threadsafe(self.broadcast_speech(entry), loop)
                    return res
                
                agent._clean_and_format = hooked_clean_and_format
                app_agent = agent.build_graph()
                
                init_state = {
                    "is_listening": True,
                    "audio_buffer": [],
                    "current_index": 0,
                    "last_speech_time": 0.0,
                    "session_file": "",
                    "latest_raw_text": "",
                    "cleaned_text": "",
                    "should_terminate": False,
                    "active_source": "mic"
                }
                app_agent.invoke(init_state)
            except Exception as e:
                print(f"[Voice Worker Error]: {e}")
            finally:
                self.is_running = False
                if not loop.is_closed():
                    asyncio.run_coroutine_threadsafe(self.broadcast_speech(
                        {"type": "recording_status", "is_recording": False}), loop)

        import threading
        self.agent_thread = threading.Thread(target=run_worker, daemon=True)
        try:
            self.agent_thread.start()
        except Exception:
            self.is_running = False
            raise
        return {"status": "starting", "message": "已启动录音监听任务，正在初始化音频通道。", "is_recording": self.is_running}

    def stop_voice_agent(self):
        with self._control_lock:
            self.is_running = False
            return {"status": "stopping" if self.agent_thread and self.agent_thread.is_alive() else "completed",
                    "message": "已停止接收新录音，正在结束当前处理。", "is_recording": False}


voice_manager = VoiceSessionManager()


@app.websocket("/ws/voice_stream")
async def websocket_voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    voice_manager.connected_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in voice_manager.connected_websockets:
            voice_manager.connected_websockets.remove(websocket)


class VoiceStartRequest(BaseModel):
    mode: str = "mic"  # mic: 麦克风；system: 系统声音/视频会议回环


@app.post("/api/voice/start")
async def start_voice_perception(req: Optional[VoiceStartRequest] = None):
    loop = asyncio.get_event_loop()
    return voice_manager.start_voice_agent(loop, mode=(req.mode if req else "mic"))


@app.post("/api/voice/stop")
async def stop_voice_perception():
    return voice_manager.stop_voice_agent()


VOICE_INITIAL_PROMPT = (
    "这是一段学术科研与技术讨论，保留专业英文术语与专有名词（如 Transformer, Attention, LLM, RAG, "
    "Python, PyTorch, LoRA, Prompt, English, China 等），中英文混合原词输出，不要将英文翻译成中文。"
)


@app.post("/api/voice/transcribe")
async def transcribe_audio_file(file: UploadFile = File(...)):
    """接收前端麦克风录制的高清音频切片，直接调用本地 OpenAI 官方原生 Whisper Large-v3 引擎转录"""
    import tempfile
    import shutil
    import torch
    from voice_langgraph_agent import WhisperEngine

    suffix = Path(file.filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        native_model = WhisperEngine.get_whisper_native_model()
        if native_model is not None:
            # 官方原生 whisper 支持直接传音频文件路径，ffmpeg 自动提取 16kHz float32
            # 方案 A: 严格 FP16 半精度 + 贪婪单分支解码 (beam_size=1)，动态显存直降 500MB~1GB
            is_cuda = next(native_model.parameters()).is_cuda
            try:
                res = native_model.transcribe(
                    tmp_path,
                    language=None,
                    task="transcribe",
                    initial_prompt=VOICE_INITIAL_PROMPT,
                    fp16=is_cuda,
                    temperature=(0.0, 0.2, 0.4),
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                )
            except torch.OutOfMemoryError:
                logger.warning("⚠️ [Whisper Large-v3] GPU 显存紧张，自动无缝降级至 CPU 极速转录通道...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cpu_model = native_model.to("cpu").float()
                WhisperEngine._whisper_native_model = cpu_model
                res = cpu_model.transcribe(
                    tmp_path,
                    language=None,
                    task="transcribe",
                    initial_prompt=VOICE_INITIAL_PROMPT,
                    fp16=False,
                    temperature=(0.0, 0.2, 0.4),
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                )

            text = res.get("text", "").strip()
            logger.info(f"🎙️ [Whisper Large-v3 方案A 原生识别成功]: {text}")
            return {"text": text, "status": "success", "engine": "whisper-large-v3-native"}
        else:
            return {"text": "", "status": "error", "message": "Whisper 原生引擎未就绪"}
    except Exception as e:
        logger.error(f"本地 Whisper 转录异常: {e}", exc_info=True)
        return {"text": "", "status": "error", "message": str(e)}
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ==========================================
# 5. 音频资源播放与动态语音合成接口
# ==========================================
class TTSSynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = "zh-CN-XiaoyiNeural"


@app.post("/api/tts/synthesize")
async def synthesize_speech_on_demand(req: TTSSynthesizeRequest):
    """按需动态合成指定文本的导师语音并返回专属音频路径"""
    from master_agent_workflow import tts_synthesis_node
    res = tts_synthesis_node({"final_answer": req.text, "tts_voice": req.voice or "zh-CN-XiaoyiNeural"})
    if res.get("tts_audio_file") and os.path.exists(res["tts_audio_file"]):
        audio_name = Path(res["tts_audio_file"]).name
        return {"audio_url": f"/api/audio/{audio_name}", "status": "success"}
    return {"audio_url": None, "status": "failed"}


@app.get("/api/audio/{filename}")
async def get_tts_audio_by_name(filename: str):
    """获取指定唯一哈希的导师语音音频文件，彻底消除跨对话声音串音"""
    if filename == "latest":
        mp3_path = BASE_DIR / "tts_response.mp3"
    else:
        mp3_path = BASE_DIR / "audio_cache" / filename
        if not mp3_path.exists():
            mp3_path = BASE_DIR / filename

    if not mp3_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(mp3_path, media_type="audio/mpeg")

# ==========================================
# 5. 天工雷达 (TianGong Academic Radar)
# ==========================================
class RadarSearchRequest(BaseModel):
    query: Optional[str] = None
    major: Optional[str] = "PHYSICS"
    sub_categories: Optional[List[str]] = None
    search_in: str = "all"  # all | title | authors | abstract
    sort_by: str = "submittedDate"  # submittedDate | relevance
    start: int = 0
    max_results: int = 15


@app.get("/api/radar/taxonomy")
async def get_radar_taxonomy():
    """获取权威 8 大领域、各学科群与 150+ 细分子学科分类全景字典"""
    from academic_taxonomy import ACADEMIC_TAXONOMY
    return {"taxonomy": ACADEMIC_TAXONOMY}


@app.post("/api/radar/search")
async def radar_search(req: RadarSearchRequest):
    """精准多学科雷达检索，提供子学科分布直方图与高精度文献卡片数据 (支持本地快照离线容灾)"""
    # 1. 尝试从本地持久化快照中先读取（如果未输入特定搜索词，且是请求第 1 页默认流）
    sub_key = req.sub_categories[0] if (req.sub_categories and len(req.sub_categories) == 1) else "all"
    is_default_browse = (not req.query or not req.query.strip()) and req.start == 0 and req.sort_by == "submittedDate"

    try:
        res = paper_searcher.search_radar(
            query=req.query,
            major=req.major,
            sub_categories=req.sub_categories,
            search_in=req.search_in,
            sort_by=req.sort_by,
            start=req.start,
            max_results=req.max_results,
        )
        if res and res.get("papers") and is_default_browse and not res.get("is_cached_snapshot"):
            # 自动沉淀为本地快照
            radar_sync_engine.save_snapshot(req.major or "CS", sub_key, res)

        # An empty announcement batch or an exhausted page is a valid response;
        # replacing it with an old first-page snapshot would fabricate updates.
        updates_handled = res and res.get("update_status") in ("live", "no_announcements", "stale_snapshot")
        if not (res and res.get("papers")) and not updates_handled:
            cached_snapshot = radar_sync_engine.read_snapshot(req.major or "CS", sub_key)
            if cached_snapshot:
                cached_snapshot["is_cached_snapshot"] = True
                return {"status": "success", "data": cached_snapshot, "is_cached_snapshot": True}

        return {"status": "success", "data": res}
    except Exception as e:
        import traceback
        traceback.print_exc()
        # 2. 如果网络波动/超时，自动回退到本地持久化快照
        cached_snapshot = radar_sync_engine.read_snapshot(req.major or "CS", sub_key)
        if cached_snapshot:
            cached_snapshot["is_cached_snapshot"] = True
            return {"status": "success", "data": cached_snapshot, "is_cached_snapshot": True}
        raise HTTPException(status_code=500, detail=f"雷达检索异常: {str(e)}")


class RadarIngestRequest(BaseModel):
    paper_id: str
    title: str
    pdf_url: str
    authors: Optional[List[str]] = []
    year: Optional[int] = None
    summary: Optional[str] = ""


@app.post("/api/radar/ingest")
async def radar_ingest_paper(req: RadarIngestRequest):
    """将雷达发现的文献直接下载并向量化嵌入本地琅嬛书阁知识库 (BGE-M3/BAAI)，异步线程池调度避免阻塞事件循环"""
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        ingest_res = await loop.run_in_executor(
            None,
            lambda: paper_searcher.download_and_ingest(
                paper={
                    "paper_id": req.paper_id,
                    "title": req.title,
                    "pdf_url": req.pdf_url,
                    "authors": req.authors,
                    "year": req.year,
                    "summary": req.summary,
                },
                rag_engine=rag_engine,
            )
        )
        return {"status": "success", "result": ingest_res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文献入库失败: {str(e)}")


class RadarTranslateRequest(BaseModel):
    title: str
    paper_id: Optional[str] = ""
    pdf_url: Optional[str] = ""
    text: Optional[str] = ""
    abstract: Optional[str] = ""


PAPER_TRANSLATION_CACHE_DIR = UPLOAD_DIR / ".paper_translation_cache"
PAPER_TRANSLATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_translation_stem(req: RadarTranslateRequest) -> str:
    identity = (req.paper_id or req.title or "paper").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", identity).strip("._")[:100]
    return safe or hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _extract_paper_pdf_text(req: RadarTranslateRequest) -> str:
    """按需下载并提取 PDF 全文；缓存只写入被 .gitignore 忽略的 uploads 目录。"""
    stem = _safe_translation_stem(req)
    text_path = PAPER_TRANSLATION_CACHE_DIR / f"{stem}.txt"
    if text_path.exists() and text_path.stat().st_size > 0:
        return text_path.read_text(encoding="utf-8")

    pdf_path = UPLOAD_DIR / f".paper_translation_{stem}.pdf"
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        pdf_url = (req.pdf_url or "").strip()
        if not pdf_url.startswith(("http://", "https://")):
            return (req.abstract or req.text or "").strip()
        request = urllib.request.Request(
            pdf_url,
            headers={"User-Agent": "Mozilla/5.0 ResearchScholar/1.0"},
        )
        with urllib.request.urlopen(request, timeout=45) as response, pdf_path.open("wb") as output:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > 80 * 1024 * 1024:
                    raise ValueError("论文 PDF 超过 80 MB，暂不支持在线全文翻译")
                output.write(chunk)

    import pymupdf

    pages = []
    with pymupdf.open(str(pdf_path)) as document:
        for page_number, page in enumerate(document, start=1):
            # 优先采用 blocks 版面分块提取，避免双栏互相穿插
            blocks = page.get_text("blocks")
            # 过滤非文本块，并按 (y0, x0) 几何版面有序阅读流排序
            blocks = [b for b in blocks if len(b) >= 5 and b[4].strip()]
            # 识别双栏：如果 x0 跨度较大，先按栏分组
            page_blocks_text = []
            for b in blocks:
                block_txt = b[4].strip()
                # 修复连字符折断单词 (如 "algo-\nrithm" -> "algorithm")
                block_txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", block_txt)
                page_blocks_text.append(block_txt)
            
            page_text = "\n\n".join(page_blocks_text).strip()
            if page_text:
                pages.append(f"【第 {page_number} 页】\n{page_text}")
    full_text = "\n\n".join(pages).strip()
    if not full_text:
        full_text = (req.abstract or req.text or "").strip()
    if full_text:
        text_path.write_text(full_text, encoding="utf-8")
    return full_text


def _split_translation_chunks(text: str, max_chars: int = 3800) -> List[str]:
    """智能将长篇论文文本按段落切分，单块保持在 3500~4000 字符内，规避单次生成超时并提高并发吞吐"""
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars)] or [paragraph]
        for piece in pieces:
            if current and len(current) + len(piece) + 2 > max_chars:
                chunks.append(current)
                current = ""
            current = f"{current}\n\n{piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def _translate_full_paper(req: RadarTranslateRequest) -> Dict[str, Any]:
    from openai import OpenAI
    from config_utils import get_translation_llm_client_kwargs, get_translation_llm_model

    translation_cache_path = PAPER_TRANSLATION_CACHE_DIR / f"{_safe_translation_stem(req)}.json"
    cached_data: Dict[str, Any] = {}
    if translation_cache_path.exists() and translation_cache_path.stat().st_size > 0:
        try:
            cached = json.loads(translation_cache_path.read_text(encoding="utf-8"))
            if cached.get("version") == 4 and cached.get("full_text_zh"):
                has_failures = "网络请求超时，已安全呈现原文" in cached["full_text_zh"]
                if not has_failures:
                    return cached
                cached_data = cached
        except (OSError, json.JSONDecodeError):
            pass

    client_kwargs = get_translation_llm_client_kwargs()
    if not client_kwargs.get("api_key"):
        raise ValueError("未配置 API Key")
    source_text = _extract_paper_pdf_text(req)
    if not source_text:
        raise ValueError("无法提取论文正文，请检查 PDF 链接是否可访问")

    client = OpenAI(**client_kwargs)
    model = get_translation_llm_model()
    if cached_data.get("abstract_zh") and cached_data.get("thesis_zh"):
        summary_data = {
            "abstract_zh": cached_data["abstract_zh"],
            "thesis_zh": cached_data["thesis_zh"],
        }
    else:
        summary_prompt = f"""请为下面这篇学术论文提供中文阅读预览。将英文摘要准确翻译成中文，并根据给出的核心论断整理1~2句中文核心解读。
不得补充原文没有的信息，保留专业术语、缩写和数学符号。严格输出 JSON：
{{"abstract_zh": "中文摘要翻译", "thesis_zh": "中文核心解读"}}

论文标题：{req.title}
核心论断：{req.text or ""}
英文摘要：{req.abstract or ""}"""
        summary_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": summary_prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=120,
        )
        try:
            summary_data = _parse_llm_json_object(summary_response.choices[0].message.content or "")
        except Exception:
            summary_data = {}
    chunks = _split_translation_chunks(source_text)
    translated_chunks: List[str] = [""] * len(chunks)
    cached_chunks = cached_data.get("chunks_zh")
    if isinstance(cached_chunks, list) and len(cached_chunks) == len(chunks):
        for i, val in enumerate(cached_chunks):
            if isinstance(val, str) and val.strip() and "网络请求超时，已安全呈现原文" not in val:
                translated_chunks[i] = val
    elif cached_data.get("full_text_zh") and int(cached_data.get("source_chunks") or 0) == len(chunks):
        # Legacy cache fallback: preserve prefix before the first failed chunk marker.
        m_fail = re.search(r"> ⚠️ 【第 (\d+) 卷网络请求超时，已安全呈现原文】", cached_data["full_text_zh"])
        if m_fail:
            first_failed_idx = int(m_fail.group(1))
            prefix_zh = cached_data["full_text_zh"][:m_fail.start()].strip()
            if first_failed_idx > 1 and prefix_zh:
                translated_chunks[0] = prefix_zh
                for i in range(1, min(first_failed_idx - 1, len(chunks))):
                    translated_chunks[i] = "__MERGED_IN_PREFIX__"

    def _translate_single_chunk(task_info):
        idx, chunk_content = task_info
        chunk_prompt = f"""你是一位严谨资深的学术学者与翻译专家。请将以下论文正文（第 {idx}/{len(chunks)} 卷）翻译为高质量、通顺连贯的中文学术译文。

【排版与数学公式严格规范（必须遵守）】：
1. 标准学术 Markdown 排版：一级章节使用 # 标题，二级小节使用 ## 标题，段落与段落之间保留空行。
2. 修复 PDF 破碎断行：自动合并被 PDF 分栏或换行打断的不完整句子，还原为结构清晰、通畅连贯的自然学术段落；图表内的零散标签请适度整理或以列表呈现，严禁输出毫无意义的孤立垂直断行。
3. 严格规范数学公式与变量（核心）：所有数学变量、符号、行内公式必须统一使用标准 LaTeX 并严格包裹在 $...$ 中（如 $H = 3.8$, $c'$, $c^{{\\sharp}}$, $\\mathcal{{H}}$）；独立块级大公式必须使用 $$...$$ 单独成段包裹，严禁输出没有 LaTeX 标记的散装乱码。
4. 保持学术严谨与专有名词：保留专业缩写、算法代码、文献引用标号（如 [1]、[2]）和页码标记（如【第 X 页】）；中英混合时保留技术术语与专有名词（如 Transformer, LoRA, BRIDGE, BOS 等）原词输出，不强行翻译。

论文标题：{req.title}

原文内容：
{chunk_content}

请严格输出 JSON：{{"translation": "符合上述 Markdown 与 LaTeX 规范的完整中文译文"}}"""
        for retry in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": chunk_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    timeout=180,
                )
                c = resp.choices[0].message.content or ""
                try:
                    t_str = json.loads(c).get("translation", "").strip()
                except json.JSONDecodeError:
                    t_str = c.strip()
                if t_str:
                    return idx, t_str
            except Exception as e:
                sleep_sec = 2.0 * (retry + 1)
                print(f"⚠️ [TranslateChunk Note] 第 {idx}/{len(chunks)} 卷翻译尝试 {retry+1} 异常 ({e})，等待 {sleep_sec}s 重试...")
                time.sleep(sleep_sec)
        return idx, f"> ⚠️ 【第 {idx} 卷网络请求超时，已安全呈现原文】\n\n{chunk_content}"

    # 🚀 并发翻译多个适度分段 (仅翻译尚未成功的分段，实现断点续译)
    from concurrent.futures import ThreadPoolExecutor
    tasks = [(i, c) for i, c in enumerate(chunks, start=1) if not translated_chunks[i - 1]]
    if tasks:
        with ThreadPoolExecutor(max_workers=min(len(tasks), 3)) as pool:
            for idx, trans_text in pool.map(_translate_single_chunk, tasks):
                translated_chunks[idx - 1] = trans_text

    final_chunks = [c for c in translated_chunks if c != "__MERGED_IN_PREFIX__"]
    full_text_zh = "\n\n".join(final_chunks).strip()
    result = {
        "version": 4,
        "title_zh": req.title,
        "thesis_zh": summary_data.get("thesis_zh", "").strip(),
        "abstract_zh": summary_data.get("abstract_zh", "").strip(),
        "full_text_zh": full_text_zh,
        "chunks_zh": translated_chunks,
        "source_chunks": len(chunks),
    }
    translation_cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def _parse_llm_json_object(raw_text: str) -> Dict[str, Any]:
    """Extract a JSON object from LLM output, tolerating ```json fences and unescaped LaTeX backslashes."""
    text = str(raw_text or "").strip()
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()

    def _fix_latex_escapes(s: str) -> str:
        # Double single backslashes not forming valid JSON escapes (e.g. \varepsilon, \sqrt, \Omega)
        return re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', s)

    candidates = [text, _fix_latex_escapes(text)]
    start = text.find("{")
    if start != -1:
        sub = text[start:]
        sub_fixed = _fix_latex_escapes(sub)
        candidates.extend([
            sub,
            sub_fixed,
            re.sub(r",\s*([}\]])", r"\1", sub_fixed),
        ])

    decoder = json.JSONDecoder()
    for cand in candidates:
        for strict in (True, False):
            try:
                parsed, _ = decoder.raw_decode(cand) if strict else (json.loads(cand, strict=False), 0)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    raise ValueError(f"模型未返回有效 JSON（前 120 字：{text[:120]}）")


def _translate_summary(req: RadarTranslateRequest) -> Dict[str, Any]:
    from openai import OpenAI
    from config_utils import get_llm_timeout, get_translation_llm_client_kwargs, get_translation_llm_model

    client_kwargs = get_translation_llm_client_kwargs()
    if not client_kwargs.get("api_key"):
        raise ValueError("未配置 API Key")
    client = OpenAI(**client_kwargs)
    prompt = f"""你是一位严谨资深的学术学者与翻译专家。请针对以下论文生成中文 AI 摘要与精译。
请准确翻译标题和英文摘要，并用1~2句概括核心方法、解决的问题与创新贡献。不得臆造信息，保留专业术语、缩写、公式和变量。
请严格按 JSON 输出：
{{"title_zh":"中文学术标题","thesis_zh":"中文核心创新与突破提要","abstract_zh":"中文摘要精译"}}

论文英文标题：{req.title}
论文英文摘要：{(req.abstract or req.text or "").strip()}"""
    response = client.chat.completions.create(
        model=get_translation_llm_model(),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
        timeout=get_llm_timeout(60.0),
    )
    return _parse_llm_json_object(response.choices[0].message.content or "")


@app.post("/api/radar/translate")
async def radar_translate(req: RadarTranslateRequest):
    """原有的 AI 摘要与精译入口；只在用户点击后调用。"""
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _translate_summary, req)
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/radar/translate_full")
async def radar_translate_full(req: RadarTranslateRequest):
    """独立的整篇论文翻译入口；只在用户点击全文按钮后调用。"""
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _translate_full_paper, req)
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ==========================================
# 5.1 网站图标与科研搭档静态形象服务
# ==========================================
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    icon_path = STATIC_DIR / "img" / "partner_ruoxi_scholarly_1.jpg"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/jpeg")
    return Response(content="", media_type="image/x-icon", status_code=204)


# ==========================================
# 6. 挂载前端静态文件
# ==========================================
# 静态文件挂载
img_extracted_dir = STATIC_DIR / "extracted_images"
img_extracted_dir.mkdir(parents=True, exist_ok=True)
app.mount("/extracted_images", StaticFiles(directory=str(img_extracted_dir)), name="extracted_images")
app.mount("/css", StaticFiles(directory=STATIC_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=STATIC_DIR / "js"), name="js")
app.mount("/img", StaticFiles(directory=STATIC_DIR / "img"), name="img")

if (STATIC_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def run_server():
    import uvicorn
    print("\n" + "=" * 70)
    print("🚀 【科研学习与会议协同多智能体系统】启动成功！")
    print("🌐 访问网址: http://localhost:8888")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")


if __name__ == "__main__":
    run_server()
