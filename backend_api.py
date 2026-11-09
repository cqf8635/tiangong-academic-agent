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
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("uvicorn.error")

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from document_rag_engine import UniversalDocumentRAGEngine
from intent_recognition_agent import IntentParsingError, IntentRecognitionAgent
from master_agent_workflow import MasterAgentState, MasterEngineRegistry, build_master_workflow, register_stream_callback
from llm_accounting import TrackingClient, UsageLedger
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
MasterEngineRegistry.paper_searcher = paper_searcher
master_app = build_master_workflow()


@app.on_event("startup")
async def on_app_startup():
    """应用启动挂钩：自动拉起天工雷达后台静默轮询与断点续航同步引擎，并预热音频大模型"""
    radar_sync_engine.start_scheduler()

    # 预热预加载本地 Whisper-large-v3 音频大模型至 GPU 显存，确保首次语音交互零延迟
    def warmup_whisper():
        try:
            from voice_langgraph_agent import WhisperEngine
            print("🚀 [Startup Warmup] 正在预加载本地 Whisper-large-v3 音频大模型至 GPU 显存...")
            from voice_langgraph_agent import MODEL_DIR
            if MODEL_DIR.suffix == ".pt":
                WhisperEngine.get_whisper_native_model()
            else:
                WhisperEngine.get_pipeline()
            print("✅ [Startup Warmup] 本地 Whisper-large-v3 音频模型预热完成，随时待命！")
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
    session = SessionManager.clear_session_messages(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session.model_dump()}


@app.post("/api/sessions/clear_all")
async def clear_all_sessions():
    init_s = SessionManager.clear_all_sessions()
    return {"initial_session": init_s.model_dump()}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    ok = SessionManager.delete_session(session_id)
    return {"success": ok}


# ==========================================
# 2. 聊天与问答接口 (支持多会话隔离)
# ==========================================
class ChatRequest(BaseModel):
    query: str
    image_data: Optional[str] = None  # Base64 附带截图或图片 (可为空)
    session_id: Optional[str] = None
    context_hint: Optional[str] = None
    voice: Optional[str] = "zh-CN-XiaoyiNeural"


def _execute_chat_workflow(req: ChatRequest) -> Dict[str, Any]:
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

    # 保存用户发送的消息到指定领域频道 (包含附带截图)
    user_msg = ChatMessage(sender="user", content=req.query, image_url=req.image_data)
    SessionManager.add_message(session_id, user_msg)

    recent_candidates = SessionManager.get_recent_candidates(session_id)
    initial_state: MasterAgentState = {
        "user_query": req.query,
        "user_image": req.image_data,
        "intent_decision": None,
        "retrieved_evidence": [],
        "final_answer": "",
        "target_agent": "",
        "tts_audio_file": None,
        "tts_voice": req.voice or "zh-CN-XiaoyiNeural",
        "history_summary": history_summary,
        "active_doc": active_doc,
        "topic_state": topic_state,
        "recent_paper_candidates": recent_candidates,
        "topic_catalog": session.topic_catalog if session else [],
        "matched_topic": matched_topic,
        "pending_action": None,
        "synthesized_evidence_slate": None,
        "dialectical_notes": None,
        "reflection_notes": None,
        "peer_review_notes": None,
        "review_verdict": None,
        "agent_dialogue_history": [],
        "stage_statuses": {},
        "token_usage": None,
        "status": "init",
    }

    # 动态初始化本轮 Token 消耗记账器并代理全局大模型客户端
    from llm_accounting import TrackingClient, UsageLedger
    from master_agent_workflow import MasterEngineRegistry

    round_ledger = UsageLedger()
    original_client = MasterEngineRegistry.get_llm_client()
    tracked_client = TrackingClient(original_client, round_ledger)
    MasterEngineRegistry.llm_client = tracked_client
    if MasterEngineRegistry.intent_agent is not None:
        MasterEngineRegistry.intent_agent.client = tracked_client
    if MasterEngineRegistry.paper_search_agent is not None:
        MasterEngineRegistry.paper_search_agent.llm_client = tracked_client
    if MasterEngineRegistry.meeting_summarizer is not None:
        MasterEngineRegistry.meeting_summarizer.client = tracked_client

    # 执行 LangGraph 状态图 (原生挂载 thread_id 记忆会话检查点)
    try:
        config = {"configurable": {"thread_id": session_id}}
        result_state = master_app.invoke(initial_state, config=config)
    except IntentParsingError as e:
        print(f"❌ [Intent Parsing Error]: {e}")
        failure_text = "⚠️ 意图解析未成功，本轮已停止，未执行检索或回答生成。请稍后重新发送。"
        stages = {"intent_parsing": "failed", "generation": "skipped"}
        token_usage = round_ledger.snapshot()
        SessionManager.add_message(session_id, ChatMessage(
            sender="agent", content=failure_text, stage_statuses=stages, token_usage=token_usage,
        ))
        # 错误提示只写入聊天记录，不更新主体、聚焦资料或主题结论档案。
        return {
            "session_id": session_id,
            "query": req.query,
            "intent_decision": None,
            "target_agent": "",
            "retrieved_evidence": [],
            "final_answer": failure_text,
            "mindmap_markdown": None,
            "audio_url": None,
            "agent_dialogue_history": [],
            "status": "intent_parsing_failed",
            "stage_statuses": stages,
            "token_usage": token_usage,
        }
    except Exception as e:
        print(f"❌ [Master Workflow Error]: {e}")
        stage = getattr(e, "stage", "workflow")
        failure_text = f"⚠️ 本轮处理失败，已停止（{stage}）：{e}"
        stages = {stage: "failed", "workflow": "failed"}
        token_usage = round_ledger.snapshot()
        SessionManager.add_message(session_id, ChatMessage(
            sender="agent", content=failure_text, stage_statuses=stages, token_usage=token_usage,
        ))
        # 失败直接返回，不归档主题结论，不生成导图和语音。
        return {
            "session_id": session_id,
            "query": req.query,
            "intent_decision": None,
            "target_agent": "",
            "retrieved_evidence": [],
            "final_answer": failure_text,
            "mindmap_markdown": None,
            "audio_url": None,
            "agent_dialogue_history": [],
            "status": "workflow_failed",
            "stage_statuses": stages,
            "token_usage": token_usage,
        }
    finally:
        MasterEngineRegistry.llm_client = original_client
        if MasterEngineRegistry.intent_agent is not None:
            MasterEngineRegistry.intent_agent.client = original_client
        if MasterEngineRegistry.paper_search_agent is not None:
            MasterEngineRegistry.paper_search_agent.llm_client = original_client
        if MasterEngineRegistry.meeting_summarizer is not None:
            MasterEngineRegistry.meeting_summarizer.client = original_client

    # 持久化同步主题生命周期状态机与前序学术候选池
    if result_state.get("topic_state"):
        SessionManager.update_topic_state(session_id, result_state["topic_state"])

    if result_state.get("recent_paper_candidates"):
        SessionManager.update_recent_candidates(session_id, result_state["recent_paper_candidates"])

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
    final_ans = result_state.get("final_answer", "")
    control_not_executed = (result_state.get("stage_statuses") or {}).get("system_control") == "not_executed"
    mindmap_md = None if control_not_executed else extract_answer_mindmap(final_ans, req.query)

    token_usage = round_ledger.snapshot()

    # 保存智能体回复到该领域频道
    agent_msg = ChatMessage(
        sender="agent",
        content=final_ans,
        intent_decision=result_state.get("intent_decision"),
        target_agent=result_state.get("target_agent"),
        retrieved_evidence=result_state.get("retrieved_evidence", []),
        mindmap_markdown=mindmap_md,
        audio_url=audio_url,
        agent_dialogue_history=result_state.get("agent_dialogue_history", []),
        synthesized_evidence_slate=result_state.get("synthesized_evidence_slate"),
        dialectical_notes=result_state.get("dialectical_notes") or result_state.get("peer_review_notes"),
        review_verdict=result_state.get("review_verdict"),
        stage_statuses=result_state.get("stage_statuses") or {},
        token_usage=token_usage,
    )
    SessionManager.add_message(session_id, agent_msg)
    
    # 自动归档本轮主题档案 (无损提炼核心定论，供后续轮次外科手术式调取)
    s_now = SessionManager.get_session(session_id)
    turn_num = len([m for m in (s_now.messages if s_now else []) if m.sender == "agent"])
    if not control_not_executed:
        SessionManager.record_turn_topic(session_id, turn_num, req.query, agent_msg)

    return {
        "session_id": session_id,
        "query": req.query,
        "history_summary": history_summary,
        "topic_state": result_state.get("topic_state"),
        "intent_decision": result_state.get("intent_decision"),
        "target_agent": result_state.get("target_agent"),
        "retrieved_evidence": result_state.get("retrieved_evidence", []),
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
        "agent_dialogue_history": result_state.get("agent_dialogue_history", []),
        "status": result_state.get("status"),
        "stage_statuses": result_state.get("stage_statuses") or {},
        "token_usage": token_usage,
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
            register_stream_callback(token_handler, session_id=req.session_id)
            result = await asyncio.to_thread(_execute_chat_workflow, req)
            register_stream_callback(None, session_id=req.session_id)
            await queue.put({"type": "done", "data": result})
        except Exception as e:
            register_stream_callback(None, session_id=req.session_id)
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
    if not file.filename:
        raise HTTPException(status_code=400, detail="上传文件名为空")

    clean_filename = Path(file.filename).name
    ext = Path(clean_filename).suffix.lower()
    dest_path = UPLOAD_DIR / clean_filename

    temp_suffix = f".{uuid.uuid4().hex[:8]}.upload_tmp"
    temp_file = dest_path.with_name(f".{dest_path.name}{temp_suffix}")

    # 1. 先接收至临时文件，避免上传中断或同名覆盖时直接损毁原物理文件
    try:
        with open(temp_file, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"文件接收写入临时存储失败: {e}")

    had_previous = dest_path.exists()
    backup_path = dest_path.with_name(f".{dest_path.name}.{uuid.uuid4().hex[:8]}.bak") if had_previous else None

    try:
        if had_previous and backup_path:
            shutil.copy2(str(dest_path), str(backup_path))
            try:
                rag_engine.delete_document(clean_filename)
            except Exception:
                pass

        # 原子替换正式路径
        os.replace(str(temp_file), str(dest_path))

        # 执行解析切片与入库
        chunks_count = rag_engine.ingest_file(dest_path)
        if chunks_count <= 0:
            raise ValueError("文档未生成有效切片，入库未完成")

        # 入库成功，清理备份
        if had_previous and backup_path and backup_path.exists():
            backup_path.unlink(missing_ok=True)

    except Exception as e:
        # 入库失败，清理可能残留的新切片与向量
        try:
            rag_engine.delete_document(clean_filename)
        except Exception:
            pass

        # 若此前存在原同名文件，恢复原文件并尝试重建原有索引
        if had_previous and backup_path and backup_path.exists():
            os.replace(str(backup_path), str(dest_path))
            try:
                rag_engine.ingest_file(dest_path)
            except Exception as restore_err:
                logger.error(f"恢复原文件索引失败: {restore_err}")
        else:
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)

        raise HTTPException(status_code=400, detail=f"文件解析入库失败 ({ext}): {e}")
    finally:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)

    return {
        "filename": clean_filename,
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
    return {"documents": docs}


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
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"会议记录解析失败: {e}") from e

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
        if self.is_running:
            return
        if mode not in {"mic", "system"}:
            raise ValueError(f"不支持的音源模式: {mode}")
        self.is_running = True

        def run_worker():
            try:
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
                logger.exception("实时语音处理失败，已停止")
                asyncio.run_coroutine_threadsafe(
                    self.broadcast_speech({"type": "error", "message": str(e)}), loop,
                )
            finally:
                self.is_running = False

        import threading
        self.agent_thread = threading.Thread(target=run_worker, daemon=True)
        self.agent_thread.start()

    def stop_voice_agent(self):
        self.is_running = False


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
    voice_manager.start_voice_agent(loop, mode=(req.mode if req else "mic"))
    return {"status": "started", "is_recording": True}


@app.post("/api/voice/stop")
async def stop_voice_perception():
    voice_manager.stop_voice_agent()
    return {"status": "stopped", "is_recording": False}


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

        import whisper
        samples = whisper.load_audio(tmp_path)
        text = WhisperEngine.transcribe(samples, language="auto")
        return {"text": text, "status": "success", "engine": "configured-whisper"}
    except Exception as e:
        logger.error(f"本地 Whisper 转录异常: {e}", exc_info=True)
        return {"text": "", "status": "error", "message": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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
    """精准多学科雷达检索，提供子学科分布直方图与高精度文献卡片数据"""
    # 默认浏览首屏成功后保存快照；请求失败或结果为空时不读取快照代替。
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

        return {"status": "success", "data": res}
    except Exception as e:
        import traceback
        traceback.print_exc()
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
    """提取真实 PDF 正文；摘要不能代替全文，旧版替代内容缓存不再复用。"""
    stem = _safe_translation_stem(req)
    text_path = PAPER_TRANSLATION_CACHE_DIR / f"{stem}.body-v5.txt"
    if text_path.exists():
        cached_text = text_path.read_text(encoding="utf-8").strip()
        if not cached_text:
            raise ValueError("论文正文缓存为空")
        return cached_text

    pdf_path = UPLOAD_DIR / f".paper_translation_{stem}.body-v5.pdf"
    if not pdf_path.exists():
        pdf_url = (req.pdf_url or "").strip()
        if not pdf_url.startswith(("http://", "https://")):
            raise ValueError("全文翻译缺少有效 PDF 链接，摘要不能代替正文")
        request = urllib.request.Request(
            pdf_url, headers={"User-Agent": "Mozilla/5.0 ResearchScholar/1.0"},
        )
        partial_path = pdf_path.with_suffix(".pdf.part")
        try:
            with urllib.request.urlopen(request, timeout=45) as response, partial_path.open("wb") as output:
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 80 * 1024 * 1024:
                        raise ValueError("论文 PDF 超过 80 MB，暂不支持在线全文翻译")
                    output.write(chunk)
            with partial_path.open("rb") as downloaded:
                if downloaded.read(5) != b"%PDF-":
                    raise ValueError("下载内容不是有效的 PDF")
            partial_path.replace(pdf_path)
        finally:
            partial_path.unlink(missing_ok=True)

    import pymupdf

    pages = []
    with pymupdf.open(str(pdf_path)) as document:
        for page_number, page in enumerate(document, start=1):
            blocks = page.get_text("blocks")
            blocks = [b for b in blocks if len(b) >= 7 and b[6] == 0 and b[4].strip()]
            page_blocks_text = []
            for block in blocks:
                block_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", block[4].strip())
                page_blocks_text.append(block_text)
            page_text = "\n\n".join(page_blocks_text).strip()
            if page_text:
                pages.append(f"【第 {page_number} 页】\n{page_text}")
            elif page.get_images() or page.get_drawings():
                raise ValueError(f"PDF 第 {page_number} 页含图像或图形但无法提取文字，全文提取未完成")
    full_text = "\n\n".join(pages).strip()
    if not full_text:
        raise ValueError("PDF 未提取到正文，摘要不能代替全文")
    text_path.write_text(full_text, encoding="utf-8")
    return full_text


def _translation_json(response: Any, required_fields: List[str]) -> Dict[str, Any]:
    """只有完整结束且字段齐全的模型结果才可作为翻译成功。"""
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise ValueError(f"翻译生成未完整结束: {choice.finish_reason}")
    data = json.loads(choice.message.content)
    if not isinstance(data, dict) or not all(
        isinstance(data.get(key), str) and data[key].strip() for key in required_fields
    ):
        raise ValueError("翻译结果缺少必需字段或正文为空")
    return data


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
    from config_utils import get_default_llm_model

    translation_cache_path = PAPER_TRANSLATION_CACHE_DIR / f"{_safe_translation_stem(req)}.v5.json"
    if translation_cache_path.exists():
        try:
            cached = json.loads(translation_cache_path.read_text(encoding="utf-8"))
            if cached.get("version") == 5 and cached.get("complete") is True and cached.get("full_text_zh"):
                return cached
            raise ValueError("全文翻译缓存缺少完整成功标记")
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"全文翻译缓存读取失败: {e}") from e

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise ValueError("未配置 API Key")
    source_text = _extract_paper_pdf_text(req)
    if not source_text:
        raise ValueError("无法提取论文正文，请检查 PDF 链接是否可访问")

    client = OpenAI(api_key=api_key, base_url=base_url)
    model = get_default_llm_model()
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
    summary_data = _translation_json(summary_response, ["abstract_zh", "thesis_zh"])
    chunks = _split_translation_chunks(source_text)
    translated_chunks: List[str] = [""] * len(chunks)

    def _translate_single_chunk(task_info):
        idx, chunk_content = task_info
        chunk_prompt = f"""你是一位严谨资深的学术学者与翻译专家。请将以下论文正文（第 {idx}/{len(chunks)} 卷）翻译为高质量、通顺连贯的中文学术译文。

【排版与数学公式严格规范（必须遵守）】：
1. 标准学术 Markdown 排版：一级章节使用 # 标题，二级小节使用 ## 标题，段落与段落之间保留空行。
2. 修复 PDF 破碎断行：自动合并被 PDF 分栏或换行打断的不完整句子，还原为结构清晰、通畅连贯的自然学术段落；图表内的零散标签适度整理或以列表呈现，保持段落语义完整连贯。
3. 规范数学符号与公式（核心）：所有数学变量、符号、行内公式统一使用标准 LaTeX 并包裹在 $...$ 中（如 $H = 3.8$, $c'$, $c^{{\\sharp}}$, $\\mathcal{{H}}$）；独立块级大公式使用 $$...$$ 单独成段包裹，确保渲染清晰规范。
4. 保持学术严谨与专有名词：保留专业缩写、算法代码、文献引用标号（如 [1]、[2]）和页码标记（如【第 X 页】）；中英混合时保留技术术语与专有名词（如 Transformer, LoRA, BRIDGE, BOS 等）原词输出，不强行直译。

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
                data = _translation_json(resp, ["translation"])
                return idx, data["translation"].strip()
            except Exception as e:
                sleep_sec = 2.0 * (retry + 1)
                print(f"⚠️ [TranslateChunk Note] 第 {idx}/{len(chunks)} 卷翻译尝试 {retry+1} 异常 ({e})，等待 {sleep_sec}s 重试...")
                time.sleep(sleep_sec)
        raise RuntimeError(f"第 {idx}/{len(chunks)} 卷翻译失败，全文未完成，未用原文替代译文")

    # 🚀 并发翻译多个适度分段 (并发 3 线程)，大幅缩减总等待时长，杜绝单次请求超时
    from concurrent.futures import ThreadPoolExecutor
    if not chunks:
        raise ValueError("未取得可翻译的论文正文")
    tasks = [(i, c) for i, c in enumerate(chunks, start=1)]
    with ThreadPoolExecutor(max_workers=min(len(chunks), 3)) as pool:
        for idx, trans_text in pool.map(_translate_single_chunk, tasks):
            translated_chunks[idx - 1] = trans_text

    full_text_zh = "\n\n".join(translated_chunks).strip()
    result = {
        "version": 5,
        "complete": True,
        "title_zh": req.title,
        "thesis_zh": summary_data.get("thesis_zh", "").strip(),
        "abstract_zh": summary_data.get("abstract_zh", "").strip(),
        "full_text_zh": full_text_zh,
        "source_chunks": len(chunks),
    }
    translation_cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def _translate_summary(req: RadarTranslateRequest) -> Dict[str, Any]:
    from openai import OpenAI
    from config_utils import get_default_llm_model

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未配置 API Key")
    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))
    prompt = f"""你是一位严谨资深的学术学者与翻译专家。请针对以下论文生成中文 AI 摘要与精译。
请准确翻译标题和英文摘要，并用1~2句概括核心方法、解决的问题与创新贡献。不得臆造信息，保留专业术语、缩写、公式和变量。
请严格按 JSON 输出：
{{"title_zh":"中文学术标题","thesis_zh":"中文核心创新与突破提要","abstract_zh":"中文摘要精译"}}

论文英文标题：{req.title}
论文英文摘要：{(req.abstract or req.text or "").strip()}"""
    response = client.chat.completions.create(
        model=get_default_llm_model(),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
        timeout=25,
    )
    return _translation_json(response, ["title_zh", "thesis_zh", "abstract_zh"])


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
    print("🌐 访问网址: http://localhost:8008")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")


if __name__ == "__main__":
    run_server()
